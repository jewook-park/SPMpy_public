# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
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

# %%

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
):
    """
    Plateau (terrace) based tilt removal for 2D STM images (sxm).
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

        out_da = xr.DataArray(Z_corr, coords=da.coords, dims=da.dims, attrs=da.attrs)

        if overwrite:
            out[var] = out_da
        else:
            out[f"{var}_plateautilt"] = out_da

        out.attrs[f"{var}_plateau_tilt"] = {
            "a_avg": a_avg,
            "b_avg": b_avg,
            "n_plateaus": int(len(areas)),
            "grad_sigma": float(grad_sigma),
            "grad_threshold": float(grad_thr),
            "min_plateau_area": int(min_plateau_area),
        }

    return out

