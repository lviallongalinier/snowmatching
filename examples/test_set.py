#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file present an example of the use of DTW_set function
on a set of randomly generated profiles.
"""

import os.path
import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage

import snowmatching.DTW.usefull

_here = os.path.dirname(os.path.realpath(__file__))
profile_filename = os.path.join(_here, 'profiles.csv')

profiles = np.genfromtxt(profile_filename)

profile = profiles[:, 0]
depth = np.arange(profile.size)
depth_grid = np.arange(0, depth.max(), 0.1)
depth_n = np.ones(profile.size)

# A set of profiles is generated from the reference profile `profile`
# by perturbing both depth and value
profiles = []
for i in range(20):
    dist = np.abs(np.random.normal(loc=0.6, scale=0.5, size=depth.size))
    depth_new = np.cumsum((depth_n * dist) / dist.mean())
    value_move = np.random.normal(scale=1, size=profile.size)

    # Sampling
    profile_ = snowmatching.DTW.usefull.oversample_profile(depth_new, profile + value_move, depth_grid, is_sorted=True)
    profiles.append(profile_[np.newaxis, :])

# Actually average the generated profiles
# with use of DTW_set
coeffs = np.array([[1, 1, 0]])
profiles = np.array(profiles)

print('Input values shape')
print('pr', profiles.shape)
print('de', depth_grid.shape)
print('co', coeffs.shape)

t_ = time.time()
M, depth_moved_s, value_moved_s = snowmatching.DTW.usefull.autofit_set(profiles, depth_grid, coeffs)
print('DTW_set in {:.5f}s'.format(time.time() - t_))

print('Output values shape')
print('M ', M.shape)
print('dm', depth_moved_s.shape)
print('vm', value_moved_s.shape)

# Plots
fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, sharex=True, sharey=True)

ax0.step(profile, depth, color='k')
ax0.title.set_text('Reference')

for i in range(profiles.shape[0]):
    ax1.step(profiles[i, 0, :], depth_grid)
    ax2.step(profiles[i, 0, :], depth_moved_s[i, :], color='gray')

ax2.step(M[0, :], depth_grid, color='red')

M_ = scipy.ndimage.gaussian_filter(M[0, :], 1)
ax3.step(profile, depth, color='k')
ax3.step(M_, depth_grid, color='red')

ax1.title.set_text('Set of profiles')
ax2.title.set_text('Mean')
ax3.title.set_text('Comparison')
plt.show()
