#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path

import numpy as np

import snowmatching.DTW.usefull

_here = os.path.dirname(os.path.realpath(__file__))
profile_filename = os.path.join(_here, 'profiles.csv')

profiles = np.genfromtxt(profile_filename)

profile_ref = profiles[:, 0]
profile_tomatch = profiles[:, 1]
depth = np.arange(profile_ref.size)

# Sampling
depth_grid = np.arange(0, depth.max(), 0.1)
profile_ref = snowmatching.DTW.usefull.oversample_profile(depth, profile_ref, depth_grid, is_sorted=True)
profile_tomatch = snowmatching.DTW.usefull.oversample_profile(depth, profile_tomatch, depth_grid, is_sorted=True)

# Depth grid have shape (N)
# profiles should have shape (1, N)
profile_ref = profile_ref[np.newaxis, :]
profile_tomatch = profile_tomatch[np.newaxis, :]

coeffs = np.array([[1, 1, 0]])

print('pr', profile_ref.shape)
print('pr', profile_tomatch.shape)
print('co', coeffs.shape)


sigma_moved, depth_moved = snowmatching.DTW.usefull.fit_to_ref_multi(profile_tomatch, profile_ref, coeffs, depth_grid, 0)

print('sm', sigma_moved.shape)
print('dm', depth_moved.shape)

import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True)
ax1.step(profile_ref[0, :], depth_grid, label='Reference')
ax1.title.set_text('Reference')
ax2.step(profile_tomatch[0, :], depth_grid, label='Profile to match')
ax2.title.set_text('To match')
ax3.step(profile_tomatch[0, :], depth_moved[1, :], label='Profile matched')
ax3.title.set_text('Matched')
plt.show()
