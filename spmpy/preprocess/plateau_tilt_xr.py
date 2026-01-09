# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

#
# # Plateau-based Tilt Removal for STM (sxm images)
#
# This notebook provides a **physically motivated tilt-correction method** for
# 2D STM images (`.sxm`), based on **automatic plateau (terrace) detection**.
#
# Unlike generic plane fitting, this approach:
# - Uses **terraces as physical references**
# - Removes only the **average global tilt**
# - **Preserves step heights and absolute offsets**
#
# ---
#
# ## Conceptual Overview
#
# 1. **Noise suppression**  
#    A Gaussian filter is applied before gradient computation.
#
# 2. **Step edge detection**  
#    The gradient magnitude \(|∇z|\) is used to identify step edges.
#
# 3. **Plateau (terrace) identification**  
#    Low-gradient regions are treated as plateaus.
#    Connected-component labeling separates individual terraces.
#
# 4. **Local plane fitting on plateaus**  
#    For each plateau:
#    \[
#    z(x,y) ≈ a_i x + b_i y + c_i
#    \]
#    Only the slope terms \((a_i, b_i)\) are retained.
#
# 5. **Area-weighted averaging**  
#    The global tilt is estimated as:
#    \[
#    a_{avg} = \frac{\sum A_i a_i}{\sum A_i}, \quad
#    b_{avg} = \frac{\sum A_i b_i}{\sum A_i}
#    \]
#
# 6. **Tilt removal**  
#    Only the averaged tilt plane is removed:
#    \[
#    z_{corr}(x,y) = z(x,y) - (a_{avg} x + b_{avg} y)
#    \]
#    The height offset is preserved.
#
# ---
#
# ## Design Principles
#
# - **2D STM images only** (sxm-style, no grid / bias dimension)
# - **Mask support**
#   - `mask == True`  → included in plateau detection and fitting
#   - `mask == False` → excluded
# - **API consistency** with `plane_fit_xr`
# - **Safe defaults**
#   - `ch='all'`
#   - `mask=None`
#   - `overwrite=False`
#
# ---
#
# ## Typical Usage
#
# ```python
# ds_corr = plateau_tilt_xr(
#     ds_sxm,
#     ch="Z_fwd",
#     grad_sigma=1.0,
#     min_plateau_area=300,
# )
# ```
#
# ### Using a reference mask
# ```python
# ds_corr = plateau_tilt_xr(
#     ds_sxm,
#     ch="Z_fwd",
#     mask=terrace_mask,
# )
# ```
#
# ### Overwriting the original channel
# ```python
# ds_corr = plateau_tilt_xr(
#     ds_sxm,
#     ch="Z_fwd",
#     overwrite=True,
# )
# ```
#

# +
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter, label


def plateau_tilt_xr(
    ds: xr.Dataset,
    ch: str = "all",
    grad_sigma: float = 1.0,
    grad_threshold: float | None = None,
    min_plateau_area: int = 200,
    mask: np.ndarray | None = None,
    overwrite: bool = False,
    store_plateau_mask: bool = False,
):
    """
    Plateau (terrace) based tilt removal for 2D STM images (sxm).
    
    This function estimates and removes a *global linear tilt* from 2D STM data
    by identifying plateau (terrace) regions separated by step edges and
    averaging the slopes fitted on those plateaus.
    
    Notes
    -----
    - Only the global linear tilt (a*x + b*y) is removed.
      Absolute height offsets and physical step heights between plateaus
      are intentionally preserved.
    
    - Plateau detection is *gradient-based* and relies on Gaussian smoothing
      (`grad_sigma`) prior to step-edge detection.
    
    - If plateau fitting fails or produces unexpected results
      (e.g. step offsets appear to be removed or distorted),
      the most common cause is *failed plateau segmentation* due to
      insufficient smoothing.
    
      In such cases, users are strongly encouraged to:
        1) Enable `store_plateau_mask=True` to inspect detected plateau regions.
        2) Increase `grad_sigma` gradually so that step edges are clearly
           separated in the smoothed image.
        3) Re-run the tilt correction after confirming that distinct plateaus
           are properly segmented.
    
    - The algorithm is intentionally conservative:
      it does not attempt to automatically "fix" ambiguous segmentation,
      but instead exposes the segmentation result for user inspection
      and controlled parameter tuning.
    """


    if not isinstance(ds, xr.Dataset):
        raise TypeError("Input must be an xarray.Dataset")

    if ch == "all":
        ch_list = list(ds.data_vars)
    else:
        if ch not in ds.data_vars:
            raise ValueError(f"Channel '{ch}' not found")
        ch_list = [ch]

    out = ds.copy()

    for var in ch_list:
        da = ds[var]
        if da.ndim != 2:
            continue

        y_dim, x_dim = da.dims
        x = da[x_dim].values
        y = da[y_dim].values
        Z = da.values.astype(float)

        if mask is not None and mask.shape != Z.shape:
            raise ValueError("mask must have the same shape as the image")

        # -------------------------------------------------
        # Smoothing + gradient
        # -------------------------------------------------
        Z_smooth = gaussian_filter(Z, sigma=float(grad_sigma))
        dZdy, dZdx = np.gradient(Z_smooth, y, x)
        grad = np.sqrt(dZdx**2 + dZdy**2)

        if grad_threshold is None:
            grad_thr = np.nanmedian(grad) + 2.0 * np.nanstd(grad)
        else:
            grad_thr = float(grad_threshold)

        step_mask = grad > grad_thr
        plateau_mask = ~step_mask

        if mask is not None:
            plateau_mask &= mask

        labels, n_labels = label(plateau_mask)
        Xg, Yg = np.meshgrid(x, y)

        slopes = []
        areas = []

        # -------------------------------------------------
        # Plane fitting per plateau
        # -------------------------------------------------
        for lab in range(1, n_labels + 1):
            region = labels == lab
            area = int(np.count_nonzero(region))
            if area < min_plateau_area:
                continue

            xr_p = Xg[region]
            yr_p = Yg[region]
            zr_p = Z[region]

            A = np.column_stack([xr_p, yr_p, np.ones_like(xr_p)])
            coeff, _, _, _ = np.linalg.lstsq(A, zr_p, rcond=None)
            a_i, b_i, _ = coeff

            slopes.append((float(a_i), float(b_i)))
            areas.append(area)

        if not slopes:
            raise RuntimeError("No valid plateaus detected.")

        slopes = np.asarray(slopes)
        areas = np.asarray(areas)

        a_avg = float(np.average(slopes[:, 0], weights=areas))
        b_avg = float(np.average(slopes[:, 1], weights=areas))

        tilt_plane = a_avg * Xg + b_avg * Yg
        Z_corr = Z - tilt_plane

        out_da = xr.DataArray(
            Z_corr,
            coords=da.coords,
            dims=da.dims,
            attrs=da.attrs,
        )

        if overwrite:
            out[var] = out_da
        else:
            out[f"{var}_plateautilt"] = out_da

        # -------------------------------------------------
        # NetCDF-safe plateau attrs (unchanged)
        # -------------------------------------------------
        out.attrs[f"{var}_plateau_tilt_a_avg"] = float(a_avg)
        out.attrs[f"{var}_plateau_tilt_b_avg"] = float(b_avg)
        out.attrs[f"{var}_plateau_tilt_n_plateaus"] = int(len(areas))
        out.attrs[f"{var}_plateau_tilt_grad_sigma"] = float(grad_sigma)
        out.attrs[f"{var}_plateau_tilt_grad_threshold"] = float(grad_thr)
        out.attrs[f"{var}_plateau_tilt_min_plateau_area"] = int(min_plateau_area)

        # -------------------------------------------------
        # [OPTIONAL] Store plateau regions for inspection
        # -------------------------------------------------
        if store_plateau_mask:
            out[f"{var}_plateau_mask"] = xr.DataArray(
                plateau_mask,
                coords=da.coords,
                dims=da.dims,
            )
            out[f"{var}_plateau_labels"] = xr.DataArray(
                labels.astype(np.int32),
                coords=da.coords,
                dims=da.dims,
            )

    return out

# -

#
# ## 🧭 Plateau Existence Decision Logic (Pre-Fit Validation)
#
# ### Motivation
# Plateau fitting should **only** be performed when a physically meaningful
# flat (plane-like) region exists in the image. Blindly applying plateau fitting
# can lead to unstable parameters and misleading results.
#
# Therefore, a **pre-fit validation step** is introduced.
#
# ---
#
# ### Plateau / Plane Detection Strategy
#
# 1. **Plane-like region detection**
#    - A plane (low-gradient region) is identified based on a local gradient
#      or residual criterion (implementation-dependent).
#    - Only pixels satisfying the plane criterion are considered
#      *plateau candidates*.
#
# 2. **Area fraction requirement**
#    - Let:
#      - `N_plateau` = number of pixels classified as plateau
#      - `N_total` = total number of pixels in the image
#    - Plateau fitting is allowed **only if**:
#      ```
#      (N_plateau / N_total) ≥ 0.10
#      ```
#      i.e. at least **10% of the total image area**.
#
# ---
#
# ### Control Flow
#
# 1. Existing preprocessing (tilt removal, background correction, etc.) runs first.
# 2. Plateau candidate region is evaluated.
# 3. A message is printed:
#    - If no valid plane is found:
#      - `"No plateau region detected — plateau fitting skipped."`
#    - If a plane is found:
#      - `"Plateau region detected: XX.X% of total area."`
# 4. Plateau fitting is executed **only if** the area threshold is satisfied.
#
# ---
#
# ### Design Principles
# - Existing plateau fitting logic is **unchanged**
# - Decision logic is **additive and explicit**
# - No side effects on downstream analysis
# - Messages are always printed before fitting
#
# This ensures transparent, reproducible, and physically meaningful plateau analysis.
#

# +

import numpy as np
import xarray as xr

def detect_plateau_region(
    data: np.ndarray,
    gradient_threshold: float,
):
    """
    Detect plane-like (plateau) regions based on gradient magnitude.

    Parameters
    ----------
    data : np.ndarray
        2D input image.
    gradient_threshold : float
        Threshold on gradient magnitude below which pixels
        are considered part of a plane.

    Returns
    -------
    plateau_mask : np.ndarray of bool
        Boolean mask indicating plateau candidate pixels.
    """
    gy, gx = np.gradient(data)
    grad_mag = np.sqrt(gx**2 + gy**2)
    plateau_mask = grad_mag < gradient_threshold
    return plateau_mask


def plateau_area_fraction(plateau_mask: np.ndarray) -> float:
    """
    Compute the fractional area occupied by the plateau region.

    Parameters
    ----------
    plateau_mask : np.ndarray of bool

    Returns
    -------
    fraction : float
        Plateau area fraction relative to the full image.
    """
    return np.count_nonzero(plateau_mask) / plateau_mask.size


def should_run_plateau_fit(
    data: np.ndarray,
    gradient_threshold: float,
    min_fraction: float = 0.10,
):
    """
    Decide whether plateau fitting should be performed.

    The decision is based on whether a plane-like region
    exists and occupies at least a minimum fraction
    of the total image area.

    Parameters
    ----------
    data : np.ndarray
        2D image after preprocessing.
    gradient_threshold : float
        Gradient magnitude threshold for plane detection.
    min_fraction : float, optional
        Minimum required plateau area fraction (default: 0.10).

    Returns
    -------
    run_fit : bool
        Whether plateau fitting should be executed.
    plateau_fraction : float
        Detected plateau area fraction.
    plateau_mask : np.ndarray of bool
        Mask of detected plateau region.
    """
    plateau_mask = detect_plateau_region(data, gradient_threshold)
    fraction = plateau_area_fraction(plateau_mask)

    if fraction == 0.0:
        print("No plateau region detected — plateau fitting skipped.")
        return False, fraction, plateau_mask

    print(
        f"Plateau region detected: {fraction * 100:.1f}% of total area."
    )

    if fraction < min_fraction:
        print(
            f"Plateau area below threshold ({min_fraction * 100:.0f}%) — fitting skipped."
        )
        return False, fraction, plateau_mask

    return True, fraction, plateau_mask


def run_plateau_fit_if_valid(
    data: np.ndarray,
    gradient_threshold: float,
    min_fraction: float,
    plateau_fit_func,
    *args,
    **kwargs,
):
    """
    Wrapper that conditionally executes plateau fitting.

    Parameters
    ----------
    data : np.ndarray
        2D image after preprocessing.
    gradient_threshold : float
        Threshold for plane detection.
    min_fraction : float
        Minimum required plateau area fraction.
    plateau_fit_func : callable
        Existing plateau fitting function.
    *args, **kwargs :
        Passed directly to `plateau_fit_func`.

    Returns
    -------
    result or None
        Output of `plateau_fit_func` if executed,
        otherwise None.
    """
    run_fit, frac, mask = should_run_plateau_fit(
        data,
        gradient_threshold,
        min_fraction,
    )

    if not run_fit:
        return None

    return plateau_fit_func(data, mask=mask, *args, **kwargs)

