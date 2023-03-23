# Matching of snow profiles

The matching presented here is designed to compare and fit snow profile against a reference profile for further comparisons. It have been originally designed by Pascal Hagenmuller and Christoph Florian Schaller ([Hagenmuller et al., 2016][Hagenmuller2016]; [Schaller et al., 2016][Schaller2016]). It uses the principles of Dynamic Time Wrapping (DTW) commonly used in audio or video fields ([Sakoe and Chiba, 1978][Sakoe1978]). It have been further used in different publications to match observed and simulated snow profiles of different resolutions, from SnowMicroPen (SMP) profiles to classical profiles as defined in [Fierz et al. (2009)][Fierz2009]. The main goal is to adjust depth in a profile to have the better alignment of layers compared to a reference profile.

The DTW method for snow profiles matching basically allows for adjusting depths of the boundaries of the layers to match two profiles or adjust the depth grid of a SMP signal to fit another one. The principle is to define a distance between two snow profiles. The distance is computed as the mean absolute difference of the parameter(s) considered in the profile:

$$ D = \frac{1}{H} \int_0^H \sum_i \frac{d_i}{\alpha_i} $$

where H is the total height of the snowpack, $i$ represents the different snow properties used for the matching (e.g. density, SSA...), $\alpha$ the normalization coefficient to make the different properties comparable or weight their relative importance and $d_i$ is the distance between the selected property at a given depth from the tested profile and the one from the reference profile (basically the absolute difference, except for grain shape).

This repository allows for a code base for matching two snow profiles with DTW method. It will progressively be improved with tools to help applying the method.

## Studies using DTW techniques for snow profiles

* [Hagenmuller et al. (2016)][Hagenmuller2016] compare SMP profiles.
* [Viallon-Galinier et al. (2021)][Viallon2021] compare observed and simulated snow profiles in terms of grain shape, grain size, handhardness, etc.
* [Herla et al. (2021)][Herla2021] provide clustering and aggregation of snow profiles.

## When to use DTW? (or when not to use it)?

Some prerequisites have to be met before applying DTW techniques.

* As the goal is to fit two snow profiles, they need to be coherent with each other. The methods is not magic, it will not manage to fit two profiles that are not sufficiently similar.
* A sufficient signal dynamic is required. Matching a homogeneous snow profile is not relevant.
* The resolution of the two profiles to match have to be of the same order of magnitude. For instance, it is not possible to directly match a SMP profile (typical resolution around or below one millimeter) and a classical profile (typical resolution above 1 cm), the SMP first have to be subsampled to around 1cm. Comparing different sources of data is possible but an under-sampling of the highly-resolved data is required first.
* Please keep in mind that DTW is a rather complex technique with underlying assumptions. Hence, if a more simple method (manual identification of layer boundaries or edge detection, for instance) is possible, it have to be preferred.

## Different steps

Different steps are required to apply the DTW method.

First, it is possible to match a partial profile, however, the main idea of the method is to adapt the depth grid of two profiles that have the same total height. Hence, it is common to first stretch the profile to have the same height as the reference profile. The most straightforward method is to apply a uniform stretching. It is also possible to constraint this step by using pre-identified layers in the profile and stretching between the identified layers.

Then, the two profiles to be compared have to share common properties. A bias (systematic error) in the profile to be matched or a higher variance could reduce the accuracy of the algorithm. It is then advised to make sure that there is no bias and a similar variance between the profiles to match.

The profiles to be compared have finally to be projected on a common depth grid. This common grid should be regular and with a relatively small interval size to allow DTW to work as corresponds to the resolution of the allowed displacements. The typical value is 1mm.

## Parameters and limits

The first approach is to match the profiles based on one parameter. The code also allowis matching based on a combination of different parameters (simultaneously, e.g. density, grain shape, liquid water content, SSA, etc., see e.g. [Viallon-Galinier et al., 2021][Viallon2021]).

The code as written allows for a maximum stretching of layers of -50% to +100%. The selected displacement is the one (with previously described constraints) that minimizes the distance between the two profiles. However, large displacement could occur for small gains in the distance as the algorithm locates the absolute minimum of the distance between the profiles. We then introduce an additional penalization term to limit large displacements of a point if it does not significantly reduce the distance. The cost which is minimized in the matching is then:

$$ C = \frac{1}{H} \int_0^H \sum_i \frac{d_i}{\alpha_i} + \frac{\|\Delta h\|}{\alpha_h}$$

with $\Delta h$ the displacement of a point at a given depth.

The value of \alpha_h$ has then to be optimized depending on the context of the matching. Possible ways for optimization are expert approach or using a value that allows displacements such as the mean total displacement is of the same order of magnitude as the typical overall snow depth difference between the two profiles.

# Technical documentation

## Installation

### Requirements

This package requires:

* Python, with version 3.8 at least
* Additional packages:
    * `Cython`
    * `numpy`
    * `scipy`
    * `matplotlib` if you plan to run the examples

Please note that the two first additional packages are required before starting the installation because they are required to build the package itself.

### Installation

You have to install this python package before use. Two methods are possible:

1. You can directly install the package with pip. Go into the package folder and run `pip install -e .`.
2. Alternatively, you can do the install manually by adding the path to this repository in your `PYTHONPATH` environment variable (`export PYTHONPATH=/path/to/snowmatching/folder`) and then manually compile the C part of this code by going into the `snowmatching/DTW` folder and typing `make` to compile the DTW C core.

## Documentation

We describe quickly hereafter the different useful steps for matching profiles. However, the provided python code is fully documented in details in the code itself. Please refer to this documentation before using the functions to be sure of how to use, and be aware of all the details.

## Preparing the data

The current code work with numpy ndarrays. If you do not directly read your data as numpy ndarrays, you first have to convert your objects into numpy ndarrays. For instance, if you use pandas, select the appropriate column and convert it into a numpy nd array (`data = df['density'].to_numpy()`).

Basically, the data is separated between the data to match (`value` array, of shape (P, N) with P the number of parameters to consider and N the number of layers or vertical axis points) and the corresponding vertical grid (depth grid `depth`).

The first preparation step is to ensure that the two profiles to match have the same total depth. This is achieved by simply rescaling the `depth` array of the profile to match. More complex rescaling can be achieved if some points are pre-identified in the profiles. No tools are provided for these special cases as each one is specific.

If necessary, the profiles could be normalized in terms of variance and mean. The function `snowmatching.DTW.usefull.scaling_profile` is provided. It computes means and variance for normalization. It should be applied on both reference and to match profiles. It remains to the user to decide whether the normalization should be done with mean and variance values computed on each profile or on a larger set of profiles to prevent introduction of biases due to discrepancies between the two profiles to match.

```python
from snowmatching.DTW import useful

normalized_reference_value = useful.scaling_profile(reference_values)
normalized_tomatch_value = useful.scaling_profile(tomatch_values)
```

The profiles have then to be projected on a common depth grid, that is used by DTW as the available displacement resolution. The typical grid resolution should be of the order of magnitude of the resolution of the highly resolved data (SMP) but lower in case of lower resolution data (e.g. classical snow profiles). The typical resolutions used up to now are around 1mm.

```python
import numpy as np

matching_depth_grid = np.arange(0, total_snow_depth, 0.001)
```

The way of projecting the data on this grid differs depending whether the data resolution is higher (or similar) to this matching grid resolution or lower.
1mm is above or around the SMP data resolution for instance. In this first case, the data have to be smoothed. Function `snowmatching.DTW.usefull.downsample_profile(depth, value, matching_depth_grid)` is provided to transfer this type of data on a matching grid (please read the full documentation as default options may not be suitable). In the second case where the data have less resolution than the matching grid, a simpler oversampling is provided by the function `snowmatching.DTW.usefull.oversample_profile(depth, value, matching_depth_grid)`.

Note that when matching profiles with significantly different resolutions (classical profiles and SMP for instance), they first have to be projected on a grid with similar resolution for the DTW algorithm to work optimally. No function is provided for this work as it is largely dependent of the type of data used.

Finally once the two profiles `reference_values` and `tomatch_values` are on the same grid `matching_depth_grid` the matching itself could be performed as follows:

```python
from snowmatching.DTW.useful import fit_to_ref_multi

moved_values, depth_grid_moved = fit_to_ref_multi(tomatch_values, reference_values, coeffs, matching_depth_grid, 0)
```

`coeffs` is an array of shape (P, 3) that defines the properties of each parameter of the profiles used for matching. If you match only one parameter (e.g. density), use the following array: `[[1, 1, 0]]`. The function return the moved value and the moved grid. Do not use simultaneously these two output values: either use the common depth grid and take the moved values or use the original values and use the new depth grid.

Some examples are provided in the `example` folder. This folder will be enriched in the future.

[Fierz2009]: https://unesdoc.unesco.org/ark:/48223/pf0000186462
[Hagenmuller2016]: https://doi.org/10.3389/feart.2016.00052
[Herla2021]: https://doi.org/10.5194/gmd-14-239-2021
[Sakoe1978]: https://doi.org/10.1109/TASSP.1978.1163055
[Schaller2016]: https://doi.org/10.5194/tc-10-1991-2016
[Viallon2021]: https://doi.org/10.1016/j.coldregions.2020.103163

# Authorship and Licence

This code is the result of the work of :

* P. Hagenmuller
* L. Viallon Galinier
* N. Calonne
* V. Nussbaum

This piece of code is provided with absolutely no warranty.

All rights are reserved, until we define a licence between the different co-authors.
