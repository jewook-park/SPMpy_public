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
# # interpolate2D_xr — Square-Pixel Interpolation for 2D STM Images
#
# This notebook documents the function **`interpolate2D_xr`**,  
# a geometry-correction utility for **2D STM (sxm) images** stored in an
# `xarray.Dataset`.
#
# The purpose of this function is to **enforce square pixels**  
# (i.e. identical physical pixel spacing along X and Y: `dx = dy`)
# while preserving the true physical scan dimensions.
#
# ---
#
# ## Purpose
#
# STM images often have different physical pixel spacings along the scan axes:
#
# - different scan ranges in X and Y
# - different pixel counts along X and Y
#
# This results in **anisotropic pixels** (`dx ≠ dy`), which can distort:
#
# - plane fitting and background removal
# - FFT and reciprocal-space analysis
# - gradient-based operations
# - quantitative line profiles
#
# `interpolate2D_xr` explicitly corrects this geometry by interpolating the image
# onto a **square-pixel grid** based on the physical scan dimensions.
#
# ---
#
# ## Physical Definition of Square Pixels
#
# Given the coordinate arrays:
#
# - `X = [x_min, …, x_max]`
# - `Y = [y_min, …, y_max]`
#
# define the physical extents and original spacings:
#
#

# %%
import numpy as np
import xarray as xr
import json


def _as_sorted_xy(ds: xr.Dataset, x_name: str = "X", y_name: str = "Y") -> xr.Dataset:
    """Return a copy sorted by X then Y to guarantee monotonic coordinates for interpolation."""
    ds2 = ds.copy()
    if x_name in ds2.coords:
        ds2 = ds2.sortby(x_name)
    if y_name in ds2.coords:
        ds2 = ds2.sortby(y_name)
    return ds2


def interpolate2D_xr(
    ds: xr.Dataset,
    ch: str = "all",
    overwrite: bool = True,
    method: str = "linear",
    x_name: str = "X",
    y_name: str = "Y",
):
    """
    Interpolate 2D STM images to enforce square pixels (dx == dy) **without introducing NaNs**.

    Key point (important)
    ---------------------
    The output Dataset MUST use the interpolated coordinates. If you interpolate variables
    and then assign them into a copy of the original Dataset (with old coords),
    xarray will align by coordinate labels and reindex, creating NaNs.

    This function avoids that by using the interpolated Dataset (`ds_interp`) as the output container.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset. Must contain coordinates `x_name` and `y_name` (defaults: 'X', 'Y').

    ch : str, default 'all'
        Channel selection.
        - 'all' : apply to all 2D variables with dims (Y, X) (in any order)
        - otherwise : apply to the named variable only

    overwrite : bool, default True
        - True  : output keeps original variable names on the new square-pixel grid.
        - False : output contains BOTH:
            * original variables on dims (Y0, X0)
            * interpolated variables on dims (Y, X) named '{var}_interp'

        This is implemented by renaming the original dataset's dims to (Y0, X0) before merging,
        so there is no coordinate collision.

    method : str, default 'linear'
        Interpolation method used by xarray (`Dataset.interp`). Common: 'linear', 'nearest'.

    x_name, y_name : str
        Names of x and y coordinates. Defaults match SPMpy sxm convention.

    Returns
    -------
    xarray.Dataset
        Interpolated dataset with dx == dy (square pixels). NetCDF-safe diagnostic metadata is stored
        in `attrs['interpolate2D']` as a JSON string.

    Notes
    -----
    - 2D-only: variables must have exactly two dims corresponding to (Y, X) (order may vary).
    - No 3D grid spectroscopy support here (bias axis etc.).
    - Coordinates are sorted before interpolation for safety.
    """

    if not isinstance(ds, xr.Dataset):
        raise TypeError("interpolate2D_xr expects an xarray.Dataset")

    if x_name not in ds.coords or y_name not in ds.coords:
        raise ValueError(f"Dataset must contain coordinates '{x_name}' and '{y_name}'")

    # Sort for monotonic coordinates (required by xarray/scipy interpolation)
    ds_sorted = _as_sorted_xy(ds, x_name=x_name, y_name=y_name)

    # Channel list
    if ch == "all":
        ch_list = list(ds_sorted.data_vars)
    else:
        if ch not in ds_sorted.data_vars:
            raise ValueError(f"Channel '{ch}' not found in Dataset")
        ch_list = [ch]

    # Physical extents and current spacings
    X = np.asarray(ds_sorted[x_name].values, dtype=float)
    Y = np.asarray(ds_sorted[y_name].values, dtype=float)

    x_min, x_max = float(X.min()), float(X.max())
    y_min, y_max = float(Y.min()), float(Y.max())

    Nx, Ny = len(X), len(Y)

    Lx = x_max - x_min
    Ly = y_max - y_min

    dx = Lx / max(Nx - 1, 1)
    dy = Ly / max(Ny - 1, 1)

    # Target spacing: choose the smaller spacing to avoid inventing detail
    d = min(dx, dy)

    Nx_new = int(np.floor(Lx / d)) + 1
    Ny_new = int(np.floor(Ly / d)) + 1

    X_new = np.linspace(x_min, x_max, Nx_new)
    Y_new = np.linspace(y_min, y_max, Ny_new)

    # Interpolate the dataset onto the new square-pixel grid
    ds_interp = ds_sorted.interp({x_name: X_new, y_name: Y_new}, method=method)

    # Build NetCDF-safe diagnostics (JSON)
    diag = dict(
        x_name=str(x_name),
        y_name=str(y_name),
        dx_original=float(dx),
        dy_original=float(dy),
        dx_new=float(abs(X_new[1] - X_new[0])) if len(X_new) > 1 else float("nan"),
        dy_new=float(abs(Y_new[1] - Y_new[0])) if len(Y_new) > 1 else float("nan"),
        Nx_new=int(Nx_new),
        Ny_new=int(Ny_new),
        method=str(method),
    )

    # Update spacing attrs if they exist (keep compatible with SPMpy conventions)
    out_interp = ds_interp.copy()
    if "X_spacing" in out_interp.attrs:
        out_interp.attrs["X_spacing"] = float(abs(X_new[1] - X_new[0])) if len(X_new) > 1 else out_interp.attrs["X_spacing"]
    if "Y_spacing" in out_interp.attrs:
        out_interp.attrs["Y_spacing"] = float(abs(Y_new[1] - Y_new[0])) if len(Y_new) > 1 else out_interp.attrs["Y_spacing"]

    out_interp.attrs["interpolate2D"] = json.dumps(diag)

    # If overwrite=True, we are done. The dataset coords already match the interpolated variables.
    if overwrite:
        # Optionally restrict to selected channels by dropping other vars (only if user didn't request all).
        # Here we keep all vars by default to match typical SPMpy workflow.
        return out_interp

    # overwrite=False: return dataset containing BOTH original and interpolated results.
    # We rename original coords/dims so they do not collide with the new (X,Y).
    ds_orig = ds_sorted.copy()

    # Rename coords and dims of the original dataset
    rename_map = {x_name: f"{x_name}0", y_name: f"{y_name}0"}
    ds_orig = ds_orig.rename(rename_map)

    # Also rename the dims for each variable that used (Y,X)
    # (rename above already handles coordinate dimension variables in xarray)

    # Keep only the requested channels in the merged output (optional)
    # Here: include all original vars, but interpolated vars are stored with _interp suffix for clarity.
    ds_interp_suffix = out_interp.copy()
    ds_interp_suffix = ds_interp_suffix.rename({var: f"{var}_interp" for var in ch_list if var in ds_interp_suffix.data_vars})

    merged = xr.merge([ds_orig, ds_interp_suffix], compat="no_conflicts")

    # Carry over diagnostics (JSON) at top-level attrs
    merged.attrs = dict(ds_sorted.attrs)
    merged.attrs["interpolate2D"] = json.dumps(diag)

    return merged

# %%
# Optional quick validation (edit the path to your local file)
# import xarray as xr, numpy as np
# ds = xr.open_dataset("ds_sxm_cu.nc")
# ds_iso = interpolate2D_xr(ds, overwrite=True)
# print("Original sizes:", dict(ds.sizes))
# print("Interpolated sizes:", dict(ds_iso.sizes))
# print("dx_new == dy_new?",
#       np.isclose(ds_iso.attrs.get("X_spacing", np.nan), ds_iso.attrs.get("Y_spacing", np.nan)))
# print("NaNs in Z_fwd:", int(np.isnan(ds_iso["Z_fwd"].values).sum()))
# ds_iso.to_netcdf("ds_iso_cu.nc")
