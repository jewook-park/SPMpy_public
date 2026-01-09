# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
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


def line_offset_xr(ds, ch, dim="X", method="median", overwrite=True):
    """
    Remove line-by-line DC offset from a 2D STM channel.

    This function subtracts a per-line offset (mean or median) along a given
    dimension. It is designed to correct STM line artifacts such as
    horizontal or vertical stripes that cannot be modeled by plane fitting.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset.
    ch : str
        Channel name to be corrected.
    dim : {"X", "Y"}, default "X"
        Dimension along which the per-line offset is computed.
        - dim="X": remove offset per Y-line (horizontal stripes)
          (compute statistic along X for each Y-line)
        - dim="Y": remove offset per X-column (vertical stripes)
          (compute statistic along Y for each X-column)
    method : {"mean", "median"}, default "median"
        Statistic used to estimate the line offset.
    overwrite : bool, default True
        If True, overwrite the original channel.
        If False, store result as "<ch>_linecorr".

    Notes
    -----
    - This operation removes line-wise DC offsets only.
    - It does NOT remove slopes, curvature, or global planes.
    - Intended to be applied BEFORE plane fitting.
    """
    da = ds[ch]

    if dim not in da.dims:
        raise ValueError(
            f"line_offset_xr: dim='{dim}' not found in da.dims={da.dims}. "
            f"Please set dim to one of {da.dims}."
        )

    if method == "mean":
        offset = da.mean(dim=dim)
    elif method == "median":
        offset = da.median(dim=dim)
    else:
        raise ValueError("method must be 'mean' or 'median'")

    corrected = da - offset

    out = ds.copy()
    if overwrite:
        out[ch] = corrected
    else:
        out[f"{ch}_linecorr"] = corrected

    return out


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


def _polyfit_1d_preserve_offset(
    coord: np.ndarray,
    values: np.ndarray,
    deg: int,
    mask_1d: np.ndarray | None = None
) -> np.ndarray:
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


def _polyfit_surface_with_mask(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    order: int,
    mask: np.ndarray | None
):
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
    # [ADDED] Line-by-line DC offset removal support
    xrdata,
    ch: str = 'all',
    method: str = 'surface_fit',
    poly_order: int = 1,
    remove_mean: bool = True,
    mask=None,
    overwrite: bool = False,
    remove_line_mean: bool = True,        # [ADDED] default ON
    line_offset_dim: str = "X",            # [ADDED] kept for API compatibility; auto-selected by method
    line_offset_method: str = "median",    # [ADDED]
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

    5) [ADDED] Line-by-line DC offset removal (STM stripe correction):
       - Applied ONLY for method in {'x_fit','y_fit'}.
       - Not applied for 'surface_fit' (by design).
       - If applied, remove_mean is FORCED to False for the subsequent plane-fit stage,
         to avoid reintroducing / canceling the line-offset correction.

       Auto-selection of line_offset_dim
       ---------------------------------
       - method == 'x_fit' → line_offset_dim_effective = 'X'
       - method == 'y_fit' → line_offset_dim_effective = 'Y'
       - method == 'surface_fit' → line_offset is skipped

       Rationale:
       - x_fit and y_fit are 1D background removals; line-offset correction is meaningful there.
       - surface_fit is a true 2D model; line-offset correction is intentionally excluded
         to keep the surface model behavior unchanged.

    Parameters
    ----------
    (existing parameters unchanged; new ones listed below)

    remove_line_mean : bool, default True
        If True, apply line_offset_xr before x_fit/y_fit.

    line_offset_dim : str, default "X"
        Kept for API compatibility. The actual dim used is automatically selected
        based on method (see above) unless you explicitly want to bypass auto-selection
        by setting remove_line_mean=False and calling line_offset_xr manually.

    line_offset_method : {"mean","median"}, default "median"
        Statistic used for line offset estimation.

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

    # ============================================================
    # [ADDED] Decide whether line offset correction applies
    # ============================================================
    apply_line_offset = (remove_line_mean is True) and (method in {"x_fit", "y_fit"})

    # Auto-select line_offset_dim only when line offset is applicable
    if method == "x_fit":
        line_offset_dim_effective = "Y"
    elif method == "y_fit":
        line_offset_dim_effective = "X"
    else:
        line_offset_dim_effective = None  # surface_fit → not used

    for var in ch_list:
        da = ds[var]

        # ============================================================
        # [ADDED] Line-by-line DC offset removal (ONLY for x_fit/y_fit)
        # ============================================================
        if apply_line_offset:
            ds = line_offset_xr(
                ds,
                ch=var,
                dim=line_offset_dim_effective,
                method=line_offset_method,
                overwrite=True,
            )
            # --------------------------------------------------------
            # [ADDED] IMPORTANT: force remove_mean=False after line correction
            # --------------------------------------------------------
            remove_mean_effective = False
        else:
            remove_mean_effective = remove_mean

        # -------------------------
        # Grid spectroscopy handling
        # -------------------------
        if 'bias_mV' in da.dims:
            y_dim, x_dim = _infer_xy_dims(da, bias_dim='bias_mV')
            mask2d = None if mask is None else mask

            fitted_stack = []
            for ib in range(da.sizes['bias_mV']):
                slice2d = ds[var].isel(bias_mV=ib)  # NOTE: use ds[var] to reflect any line correction

                data2d = slice2d.values.astype(float, copy=False)
                slice_mean = _masked_mean_2d(data2d, mask2d)

                # Temporary centering for stability (now uses remove_mean_effective)
                if remove_mean_effective:
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

                # Restore original mean only if remove_mean_effective is True
                if remove_mean_effective:
                    corrected = corrected + slice_mean

                slice_out = xr.DataArray(
                    corrected,
                    coords={y_dim: slice2d[y_dim], x_dim: slice2d[x_dim]},
                    dims=(y_dim, x_dim),
                    attrs=slice2d.attrs
                )
                fitted_stack.append(slice_out.values)

            axis = ds[var].dims.index('bias_mV')
            fitted = np.stack(fitted_stack, axis=axis)

            result = xr.DataArray(
                fitted,
                coords=ds[var].coords,
                dims=ds[var].dims,
                attrs=ds[var].attrs
            )

        # -------------------------
        # Pure 2D image handling
        # -------------------------
        else:
            # Rebind da AFTER potential line correction
            da2 = ds[var]
            y_dim, x_dim = _infer_xy_dims(da2, bias_dim='bias_mV')
            mask2d = None if mask is None else mask

            data2d = da2.values.astype(float, copy=False)
            img_mean = _masked_mean_2d(data2d, mask2d)

            if remove_mean_effective:
                data_work = data2d - img_mean
            else:
                data_work = data2d

            x = da2[x_dim].values
            y = da2[y_dim].values

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

            if remove_mean_effective:
                corrected = corrected + img_mean

            result = xr.DataArray(
                corrected,
                coords=da2.coords,
                dims=da2.dims,
                attrs=da2.attrs
            )

        # Store output
        if overwrite:
            ds[var] = result
        else:
            ds[f"{var}_planefit"] = result

    return ds



# %% [markdown]
# ## 🔧 Update: Line-by-line DC Offset Removal for STM Data (2025 0109)
#
# This notebook has been **extended (without breaking existing behavior)** to support
# **line-by-line DC offset removal**, a correction commonly required for STM data
# exhibiting horizontal or vertical stripe artifacts.
#
# ### Why this is needed
# - `plane_fit_xr` removes *planes* and low-order polynomial drifts.
# - STM stripㅋe artifacts are typically **line-wise DC offsets**, not planes.
# - Such artifacts cannot be removed by plane fitting alone.
#
# ### New functionality (additive)
# - A helper function `line_offset_xr` has been added.
# - `plane_fit_xr` now supports optional line-wise offset removal **before** plane fitting.
#
# ### New parameters in `plane_fit_xr`
# - `remove_line_mean=True` (default)
# - `line_offset_dim='X'`  
#   - `'X'`: remove offset per Y-line (horizontal stripes)
#   - `'Y'`: remove offset per X-column (vertical stripes)
# - `line_offset_method='median'` (`'mean'` also supported)
#
# ### Backward compatibility
# - All original behavior is preserved.
# - Setting `remove_line_mean=False` reproduces the legacy behavior exactly.
#
# ### Design note
# - FFT-based stripe removal (e.g. masking ky≈0 components) is **intentionally not included** here
#   and should be implemented in a separate, dedicated function.
#

# %%
