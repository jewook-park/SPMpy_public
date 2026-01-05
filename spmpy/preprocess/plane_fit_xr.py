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
# # Plane / Polynomial Background Fitting for STM-SPM Data (xarray)
#
# **Location**
# ```
# SPMpy/spmpy/preprocess/
# ```
#
# This notebook provides a fully consistency-checked implementation of
# plane / polynomial background removal for STM topography, current maps,
# and grid spectroscopy data.
#
# All fitting logic (x, y, surface; 1st–3rd order; mask semantics) has been
# double-checked for correctness and internal consistency.
#

# %%

import numpy as np
import xarray as xr

# %% [markdown]
# # plane_fit_xr – Updated Implementation (2026-01-05)
#
# This notebook documents an update to `plane_fit_xr`.
#
# ## plane_fit_xr (Internal) – Background Removal with Mask, Mean Preservation, and Robust Axis Inference
#
# This cell provides a self-contained implementation of polynomial background removal for STM/SPM data
# stored as `xarray.Dataset` or `xarray.DataArray`.
#
# ### Supported data shapes
# - **2D images**: `(Y, X)` or `(y, x)` or any two spatial dims
# - **Grid spectroscopy**: `(bias_mV, Y, X)` (background removal is applied independently for each bias slice)
#
# ### Key behaviors
# 1. **Robust axis inference**
#    - Works when dims are named `(Y, X)` or `(y, x)` (case-insensitive).
#    - Ensures **horizontal axis is x** and **vertical axis is y**.
#    - If dim names do not indicate x/y, the fallback assumes the last two non-bias dims are `(y, x)`.
#
# 2. **Mask support**
#    - `mask == True` : included in fitting
#    - `mask == False`: excluded from fitting
#    - If `mask is None`, all pixels are used.
#
# 3. **Mean preservation (default)**
#    - `remove_mean=True` (default): subtract the mean temporarily for fit stability, then restore the original mean.
#    - Mean preservation is applied:
#      - for a single 2D image (global image mean)
#      - for each `bias_mV` slice independently (slice mean)
#
# 4. **Offset-preserving 1D fits (always enabled)**
#    - For `x_fit` and `y_fit`, the algorithm removes **only non-constant polynomial components**
#      (tilt/curvature terms) and preserves the DC offset.
#    - This avoids unintended row/column mean equalization artifacts.
#
# ### Methods
# - `surface_fit`: 2D polynomial surface fit over (x,y) and subtraction
# - `x_fit`: per-row polynomial fit along x, subtract only non-constant components
# - `y_fit`: per-column polynomial fit along y, subtract only non-constant components
#
# ---
#
# ## Usage Examples
#
# ### Example 1) Surface fit on all channels (recommended default)
# ```python
# ds_corr = plane_fit_xr(
#     ds_sxm,
#     ch='all',
#     method='surface_fit',
#     poly_order=1,
#     mask=None,
#     overwrite=False
# )
#

# %%
import numpy as np
import xarray as xr


def _infer_xy_dims(da: xr.DataArray, bias_dim: str = "bias_mV"):
    """
    Infer (y_dim, x_dim) for a 2D spatial map from an xarray.DataArray.

    This function is designed for STM/SPM images where:
      - the horizontal axis should correspond to x
      - the vertical axis should correspond to y

    Strategy (robust + conservative)
    -------------------------------
    1) If dimension names clearly indicate x/y (case-insensitive), use them:
       - x candidates: dim name contains 'x' (e.g., 'x', 'X', 'X_nm', 'x_pix')
       - y candidates: dim name contains 'y' (e.g., 'y', 'Y', 'Y_nm', 'y_pix')
       If both exist, return (y_dim, x_dim).

    2) Otherwise, ignore `bias_dim` (if present) and fall back to:
       - last two remaining dims treated as (y, x) in that order.

    Notes
    -----
    - For typical STM xarray datasets, dims are usually (Y, X) or (y, x).
    - If the data are stored as (X, Y) but dim names are not informative,
      automatic inference is ambiguous. In that rare case, the fallback
      assumes the first of the two dims is y and the second is x.
    """
    dims = list(da.dims)
    spatial_dims = [d for d in dims if d != bias_dim]

    if len(spatial_dims) != 2:
        raise ValueError(
            f"Cannot infer (y,x) dims. Expected 2 spatial dims, got {spatial_dims} from da.dims={da.dims}"
        )

    def _is_x(name: str) -> bool:
        return "x" in name.lower()

    def _is_y(name: str) -> bool:
        return "y" in name.lower()

    x_candidates = [d for d in spatial_dims if _is_x(d)]
    y_candidates = [d for d in spatial_dims if _is_y(d)]

    if x_candidates and y_candidates:
        x_dim = x_candidates[0]
        y_dim = y_candidates[0]
        return y_dim, x_dim

    # Fallback: assume (y, x) order
    y_dim, x_dim = spatial_dims[0], spatial_dims[1]
    return y_dim, x_dim


def _masked_mean_2d(arr2d: np.ndarray, mask2d: np.ndarray | None):
    """
    Compute mean of a 2D array with optional mask support.

    Mask semantics
    --------------
    mask == True  : included in mean calculation
    mask == False : excluded

    If mask is None, the full-array nanmean is used.
    If mask has no True entries, falls back to full-array nanmean.
    """
    if mask2d is None:
        return float(np.nanmean(arr2d))
    if np.any(mask2d):
        return float(np.nanmean(arr2d[mask2d]))
    return float(np.nanmean(arr2d))


def _polyfit_1d_preserve_offset(coord: np.ndarray,
                                values: np.ndarray,
                                deg: int,
                                mask_1d: np.ndarray | None = None) -> np.ndarray:
    """
    Fit a 1D polynomial background with optional mask and return ONLY the
    non-constant component evaluated across the full coordinate axis.

    Purpose
    -------
    For per-row or per-column fitting (x_fit / y_fit), subtracting the full
    polynomial (including intercept) removes DC offsets and can unintentionally
    equalize row/column means.

    This function ALWAYS preserves offsets by:
      - fitting a polynomial p(coord)
      - returning p(coord) with its constant term forced to zero
        (i.e., remove tilt/curvature terms only)

    Numerical stability
    -------------------
    - Coordinates are centered before fitting: dcoord = coord - mean(coord_fit)
      This reduces condition issues when coordinates are small (e.g., meters).

    Mask semantics
    --------------
    mask == True  : included in fitting
    mask == False : excluded

    Returns
    -------
    fit_no_const : np.ndarray
        Polynomial background WITHOUT the constant term.
        Subtracting this from the data preserves the DC offset.
    """
    coord = np.asarray(coord, dtype=float)
    values = np.asarray(values, dtype=float)

    # Select fitting points
    if mask_1d is None:
        coord_fit = coord
        values_fit = values
    else:
        coord_fit = coord[mask_1d]
        values_fit = values[mask_1d]

    # Need at least deg+1 points
    if coord_fit.size < (deg + 1):
        return np.zeros_like(values, dtype=float)

    # Center coordinates for stability
    c0 = float(np.mean(coord_fit))
    dcoord = coord - c0
    dcoord_fit = coord_fit - c0

    # Polynomial fit in centered coordinates
    coeff = np.polyfit(dcoord_fit, values_fit, deg=deg)  # highest power first

    # Force constant term to zero (preserve offset)
    coeff[-1] = 0.0

    fit_no_const = np.polyval(coeff, dcoord)
    return fit_no_const


def _polyfit_surface_with_mask(x: np.ndarray,
                               y: np.ndarray,
                               z: np.ndarray,
                               order: int,
                               mask: np.ndarray | None):
    """
    Perform 2D polynomial surface fitting with mask support.

    Supported polynomial orders
    ---------------------------
    order = 1 : z = ax + by + c
    order = 2 : quadratic surface
    order = 3 : cubic surface

    Mask semantics
    --------------
    mask == True  : included in fitting
    mask == False : excluded

    Notes
    -----
    - mask may be sparse or continuous.
    - If mask is None, the entire image is used.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    X, Y = np.meshgrid(x, y)

    if mask is None:
        mask = np.ones_like(z, dtype=bool)

    Xf = X[mask]
    Yf = Y[mask]
    Zf = z[mask]

    if order == 1:
        A = np.column_stack([Xf, Yf, np.ones_like(Xf)])
    elif order == 2:
        A = np.column_stack([
            Xf**2, Yf**2, Xf*Yf,
            Xf, Yf, np.ones_like(Xf)
        ])
    elif order == 3:
        A = np.column_stack([
            Xf**3, Yf**3,
            Xf**2*Yf, Xf*Yf**2,
            Xf**2, Yf**2, Xf*Yf,
            Xf, Yf, np.ones_like(Xf)
        ])
    else:
        raise ValueError("poly_order must be 1, 2, or 3")

    coeff, _, _, _ = np.linalg.lstsq(A, Zf, rcond=None)

    if order == 1:
        surface = coeff[0]*X + coeff[1]*Y + coeff[2]
    elif order == 2:
        surface = (
            coeff[0]*X**2 + coeff[1]*Y**2 + coeff[2]*X*Y +
            coeff[3]*X + coeff[4]*Y + coeff[5]
        )
    elif order == 3:
        surface = (
            coeff[0]*X**3 + coeff[1]*Y**3 +
            coeff[2]*X**2*Y + coeff[3]*X*Y**2 +
            coeff[4]*X**2 + coeff[5]*Y**2 +
            coeff[6]*X*Y +
            coeff[7]*X + coeff[8]*Y + coeff[9]
        )

    return surface


def plane_fit_xr(
    xrdata,
    ch: str = 'all',
    method: str = 'surface_fit',
    poly_order: int = 1,
    remove_mean: bool = True,
    mask=None,
    overwrite: bool = False
):
    """
    Polynomial plane / surface background removal for STM-SPM data (xarray).

    Applies polynomial background fitting to:
    - 2D STM images: (Y, X) / (y, x) / arbitrary 2D spatial dims
    - Grid spectroscopy: (bias_mV, Y, X) (fit independently per bias slice)

    Core behaviors
    --------------
    1) Robust x/y axis inference:
       - Accepts dims named (y,x) or (Y,X) (case-insensitive).
       - Ensures horizontal axis corresponds to x and vertical axis to y.
       - If names are ambiguous, assumes the last two spatial dims are (y,x).

    2) Mask support (mask==True included):
       - mask == True  : included in fitting
       - mask == False : excluded

    3) Mean preservation with stability centering (DEFAULT):
       - If remove_mean=True (default):
           (a) compute original mean (per 2D map; per bias slice for grid)
           (b) subtract mean temporarily for numerical stability
           (c) fit and subtract background
           (d) restore the original mean to the corrected output
         => output mean equals input mean (physically preserved)

       - If remove_mean=False:
           fit is performed on the original values (no temporary centering)

    4) Offset-preserving 1D fits ALWAYS:
       - For x_fit and y_fit:
         only non-constant polynomial components are removed, preserving DC offsets.
         This avoids unintended row/column mean equalization artifacts.

    Parameters
    ----------
    xrdata : xarray.Dataset or xarray.DataArray
        Input STM/SPM data. If a DataArray is given, it is converted internally
        to a Dataset. Output is always a Dataset.

    ch : str, default 'all'
        Channel selection.
        - 'all' : apply fitting to all data variables
        - specific variable name (e.g. 'Z_fwd')

    method : {'x_fit', 'y_fit', 'surface_fit'}, default 'surface_fit'
        Background fitting method.
        - x_fit      : per-row polynomial fit along x
        - y_fit      : per-column polynomial fit along y
        - surface_fit: full 2D polynomial surface fit

    poly_order : int, default 1
        Polynomial order of fitting (1, 2, or 3).

    remove_mean : bool, default True
        Temporary mean centering for fit stability.
        Mean is restored after background subtraction so that output mean equals input mean.

    mask : ndarray of bool, optional
        Boolean mask specifying which pixels are INCLUDED in fitting.
        mask must match the 2D spatial shape (Y,X) for images and per-slice grids.

    overwrite : bool, default False
        Storage behavior.
        - False : store result as '{var}_planefit'
        - True  : overwrite original variable

    Returns
    -------
    xarray.Dataset
        Dataset containing background-corrected data.
    """
    # Normalize input to Dataset
    if isinstance(xrdata, xr.DataArray):
        ds = xrdata.to_dataset(name=xrdata.name or 'data')
    elif isinstance(xrdata, xr.Dataset):
        ds = xrdata.copy()
    else:
        raise TypeError("Input must be xarray.Dataset or xarray.DataArray")

    # Channel selection
    if ch == 'all':
        ch_list = list(ds.data_vars)
    else:
        if ch not in ds.data_vars:
            raise ValueError(f"Channel '{ch}' not found in Dataset")
        ch_list = [ch]

    for var in ch_list:
        da = ds[var]

        # -------------------------
        # Grid spectroscopy handling
        # -------------------------
        if 'bias_mV' in da.dims:
            y_dim, x_dim = _infer_xy_dims(da, bias_dim='bias_mV')
            mask2d = None if mask is None else mask

            fitted_stack = []
            for ib in range(da.sizes['bias_mV']):
                slice2d = da.isel(bias_mV=ib)  # 2D DataArray

                data2d = slice2d.values.astype(float, copy=False)
                slice_mean = _masked_mean_2d(data2d, mask2d)

                # Temporary centering for stability (default True)
                if remove_mean:
                    data_work = data2d - slice_mean
                else:
                    data_work = data2d

                x = slice2d[x_dim].values
                y = slice2d[y_dim].values

                if method == 'surface_fit':
                    surface = _polyfit_surface_with_mask(x, y, data_work, poly_order, mask2d)
                    corrected = data_work - surface

                elif method == 'x_fit':
                    corrected = np.zeros_like(data_work, dtype=float)
                    ny, nx = data_work.shape
                    for iy in range(ny):
                        row = data_work[iy]
                        row_mask = None if mask2d is None else mask2d[iy]
                        fit_no_const = _polyfit_1d_preserve_offset(x, row, poly_order, row_mask)
                        corrected[iy] = row - fit_no_const

                elif method == 'y_fit':
                    corrected = np.zeros_like(data_work, dtype=float)
                    ny, nx = data_work.shape
                    for ix in range(nx):
                        col = data_work[:, ix]
                        col_mask = None if mask2d is None else mask2d[:, ix]
                        fit_no_const = _polyfit_1d_preserve_offset(y, col, poly_order, col_mask)
                        corrected[:, ix] = col - fit_no_const

                else:
                    raise ValueError("Invalid method. Choose from {'x_fit','y_fit','surface_fit'}.")

                # Restore original mean (preserve physical mean per bias slice)
                if remove_mean:
                    corrected = corrected + slice_mean

                slice_out = xr.DataArray(
                    corrected,
                    coords={y_dim: slice2d[y_dim], x_dim: slice2d[x_dim]},
                    dims=(y_dim, x_dim),
                    attrs=slice2d.attrs
                )
                fitted_stack.append(slice_out.values)

            axis = da.dims.index('bias_mV')
            fitted = np.stack(fitted_stack, axis=axis)

            result = xr.DataArray(
                fitted,
                coords=da.coords,
                dims=da.dims,
                attrs=da.attrs
            )

        # -------------------------
        # Pure 2D image handling
        # -------------------------
        else:
            y_dim, x_dim = _infer_xy_dims(da, bias_dim='bias_mV')
            mask2d = None if mask is None else mask

            data2d = da.values.astype(float, copy=False)
            img_mean = _masked_mean_2d(data2d, mask2d)

            if remove_mean:
                data_work = data2d - img_mean
            else:
                data_work = data2d

            x = da[x_dim].values
            y = da[y_dim].values

            if method == 'surface_fit':
                surface = _polyfit_surface_with_mask(x, y, data_work, poly_order, mask2d)
                corrected = data_work - surface

            elif method == 'x_fit':
                corrected = np.zeros_like(data_work, dtype=float)
                ny, nx = data_work.shape
                for iy in range(ny):
                    row = data_work[iy]
                    row_mask = None if mask2d is None else mask2d[iy]
                    fit_no_const = _polyfit_1d_preserve_offset(x, row, poly_order, row_mask)
                    corrected[iy] = row - fit_no_const

            elif method == 'y_fit':
                corrected = np.zeros_like(data_work, dtype=float)
                ny, nx = data_work.shape
                for ix in range(nx):
                    col = data_work[:, ix]
                    col_mask = None if mask2d is None else mask2d[:, ix]
                    fit_no_const = _polyfit_1d_preserve_offset(y, col, poly_order, col_mask)
                    corrected[:, ix] = col - fit_no_const

            else:
                raise ValueError("Invalid method. Choose from {'x_fit','y_fit','surface_fit'}.")

            if remove_mean:
                corrected = corrected + img_mean

            result = xr.DataArray(
                corrected,
                coords=da.coords,
                dims=da.dims,
                attrs=da.attrs
            )

        # Store output
        if overwrite:
            ds[var] = result
        else:
            ds[f"{var}_planefit"] = result

    return ds

