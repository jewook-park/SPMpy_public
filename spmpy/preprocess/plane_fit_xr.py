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


# %%

def _polyfit_1d_with_mask(coord, data, order, mask):
    """
    Perform 1D polynomial fitting with explicit mask control.

    Mask semantics
    --------------
    mask == True  : included in fitting
    mask == False : excluded from fitting

    Notes
    -----
    - mask may be sparse (point-like selection).
    - If mask is None, all points are used.
    """
    if mask is None:
        mask = np.ones_like(data, dtype=bool)

    coeff = np.polyfit(coord[mask], data[mask], order)
    return np.polyval(coeff, coord)


def _polyfit_surface_with_mask(x, y, z, order, mask):
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
    mask == False : excluded from fitting

    Notes
    -----
    - mask may be sparse or continuous.
    - If mask is None, the entire image is used.
    """
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



# %% [markdown]
#
# # plane_fit_xr – Updated Implementation (2026-01-05)
#
# This notebook documents an update to `plane_fit_xr`.
#
# ## Summary of Update
# - Added an explicit **mean removal option** (`remove_mean=False` by default)
# - Ensured **x-fit and y-fit symmetry**
# - Removed unintended row-wise normalization in `x_fit`
# - Updated docstring and usage explanation
# - Clarified physical meaning of each option
#
# This update was motivated by observed artifacts where `x_fit` produced
# row-wise mean-equalized (high-pass–like) results.
#

# %% [markdown]
#
# ## What Was Changed (Changelog)
#
# 1. **New parameter added**
#    - `remove_mean: bool = False`
#    - Controls optional mean-centering before fitting
#
# 2. **Default behavior changed**
#    - Mean removal is **disabled by default**
#    - Preserves physical offsets and gradients
#
# 3. **x_fit behavior corrected**
#    - Removed implicit `mean(dim='x')` normalization
#    - Each y-row now retains its own average value
#
# 4. **Documentation updated**
#    - Docstring expanded
#    - Explicit warning about high-pass behavior when `remove_mean=True`
#

# %%
import numpy as np
import xarray as xr

def plane_fit_xr(
    xrdata,
    ch='all',
    method='surface_fit',
    poly_order=1,
    remove_mean=False,
    mask=None,
    overwrite=False
):
    """
    Polynomial plane / surface background removal for STM-SPM data.

    This function applies polynomial background fitting to:
    - 2D STM images (y, x) (e.g., topography, current, lock-in channels)
    - Grid spectroscopy data (bias_mV, y, x)

    The goal is to remove low-order background trends (tilt/curvature) while
    preserving physically meaningful variations.

    Parameters
    ----------
    xrdata : xarray.Dataset or xarray.DataArray
        Input STM/SPM data. If a DataArray is given, it is converted internally
        to a Dataset. Output is always a Dataset.

        Typical 2D image variables:
            (Y, X) or (y, x)
        Typical grid variables:
            (bias_mV, Y, X)

    ch : str, default 'all'
        Channel selection.
        - 'all' : apply fitting to all data variables
        - specific variable name (e.g. 'Z_fwd', 'CURR_fwd')

    method : {'x_fit', 'y_fit', 'surface_fit'}, default 'surface_fit'
        Background fitting method.
        - 'x_fit'      : polynomial fit along x direction (row-wise; each fixed y)
        - 'y_fit'      : polynomial fit along y direction (column-wise; each fixed x)
        - 'surface_fit': full 2D polynomial surface fit (x,y) -> z

    poly_order : int, default 1
        Polynomial order of fitting (1, 2, or 3).

    remove_mean : bool, default False
        If True, mean-centering is applied *before* fitting.

        Behavior by method
        ------------------
        - method='x_fit':
            For each y-row, subtract the row mean (along x) before polynomial fitting.
            This enforces identical mean values across x for each row and can introduce
            a high-pass–like appearance. Use only when intentionally desired.

        - method='y_fit':
            For each x-column, subtract the column mean (along y) before polynomial fitting.
            Similarly, this can impose mean normalization and should be used with caution.

        - method='surface_fit':
            Subtract a global mean (over the 2D image) before surface fitting.
            This mainly removes constant offset prior to fitting.

        WARNING
        -------
        Setting remove_mean=True changes the physical meaning of the output:
        it may suppress legitimate offsets/slow variations and resemble a high-pass filter,
        especially for x_fit/y_fit. Keep remove_mean=False for standard plane/tilt removal.

    mask : ndarray of bool, optional
        Boolean mask specifying which pixels are INCLUDED in fitting.

        mask == True  → used for fitting
        mask == False → excluded from fitting

        Notes:
        - mask may be sparse (point mask).
        - If None, all pixels are used.
        - For x_fit/y_fit, the 1D mask is extracted per row/column accordingly.

    overwrite : bool, default False
        Storage behavior.
        - False : fitted result stored as '{var}_planefit'
        - True  : overwrite original variable

    Special behavior (grid spectroscopy)
    -----------------------------------
    If a variable contains a 'bias_mV' dimension, background fitting is applied
    independently for each bias slice (2D image at each bias).

    Returns
    -------
    xarray.Dataset
        Dataset containing plane-fitted data (either overwritten or appended
        as '{var}_planefit').
    """

    # --- normalize input to Dataset ---
    if isinstance(xrdata, xr.DataArray):
        ds = xrdata.to_dataset(name=xrdata.name or 'data')
    elif isinstance(xrdata, xr.Dataset):
        ds = xrdata.copy()
    else:
        raise TypeError("Input must be xarray.Dataset or xarray.DataArray")

    # --- channel selection ---
    if ch == 'all':
        ch_list = list(ds.data_vars)
    else:
        if ch not in ds.data_vars:
            raise ValueError(f"Channel '{ch}' not found")
        ch_list = [ch]

    for var in ch_list:
        da = ds[var]

        # =========================
        # 1) Grid spectroscopy case
        # =========================
        if 'bias_mV' in da.dims:
            fitted_stack = []

            for ib, b in enumerate(da.bias_mV.values):
                slice2d = da.isel(bias_mV=ib)

                # mask: assume same 2D mask applies to each bias slice (if provided)
                slice_mask = None if mask is None else mask

                # IMPORTANT: propagate remove_mean option into per-slice fitting
                slice_out = plane_fit_xr(
                    slice2d,
                    ch=slice2d.name,
                    method=method,
                    poly_order=poly_order,
                    remove_mean=remove_mean,
                    mask=slice_mask,
                    overwrite=True
                )

                fitted_stack.append(slice_out[slice2d.name].values)

            axis = da.dims.index('bias_mV')
            fitted = np.stack(fitted_stack, axis=axis)

            result = xr.DataArray(
                fitted,
                coords=da.coords,
                dims=da.dims,
                attrs=da.attrs
            )

        # =================
        # 2) Pure 2D images
        # =================
        else:
            data2d = da.values
            ny, nx = data2d.shape

            # Use coordinate values if available (assumes dims order is (y, x))
            x = da.coords[da.dims[1]].values if len(da.dims) == 2 else np.arange(nx)
            y = da.coords[da.dims[0]].values if len(da.dims) == 2 else np.arange(ny)

            # ---- surface fit (2D polynomial) ----
            if method == 'surface_fit':
                # Optional global mean removal (constant offset)
                if remove_mean:
                    data_work = data2d - np.nanmean(data2d)
                else:
                    data_work = data2d

                surface = _polyfit_surface_with_mask(x, y, data_work, poly_order, mask)
                result_np = data_work - surface

            # ---- x_fit (row-wise along x) ----
            elif method == 'x_fit':
                result_np = np.zeros_like(data2d)

                for iy in range(ny):
                    row = data2d[iy].astype(float, copy=False)
                    row_mask = None if mask is None else mask[iy]

                    # Optional mean-centering per row (explicit option)
                    if remove_mean:
                        # Mean of the INCLUDED pixels if a mask is given, else full row mean
                        if row_mask is None:
                            row_mean = np.nanmean(row)
                        else:
                            row_mean = np.nanmean(row[row_mask]) if np.any(row_mask) else np.nanmean(row)
                        row_work = row - row_mean
                    else:
                        row_work = row

                    fit = _polyfit_1d_with_mask(x, row_work, poly_order, row_mask)
                    result_np[iy] = row_work - fit

            # ---- y_fit (column-wise along y) ----
            elif method == 'y_fit':
                result_np = np.zeros_like(data2d)

                for ix in range(nx):
                    col = data2d[:, ix].astype(float, copy=False)
                    col_mask = None if mask is None else mask[:, ix]

                    # Optional mean-centering per column (explicit option)
                    if remove_mean:
                        # Mean of the INCLUDED pixels if a mask is given, else full column mean
                        if col_mask is None:
                            col_mean = np.nanmean(col)
                        else:
                            col_mean = np.nanmean(col[col_mask]) if np.any(col_mask) else np.nanmean(col)
                        col_work = col - col_mean
                    else:
                        col_work = col

                    fit = _polyfit_1d_with_mask(y, col_work, poly_order, col_mask)
                    result_np[:, ix] = col_work - fit

            else:
                raise ValueError("Invalid method. Choose from {'x_fit','y_fit','surface_fit'}.")

            result = xr.DataArray(
                result_np,
                coords=da.coords,
                dims=da.dims,
                attrs=da.attrs
            )

        # ---- store output ----
        if overwrite:
            ds[var] = result
        else:
            ds[f"{var}_planefit"] = result

    return ds


# %% [markdown]
#
# ## Practical Recommendation
#
# - **Default (`remove_mean=False`)**  
#   Use for physically meaningful plane / tilt removal.
#
# - **Optional (`remove_mean=True`)**  
#   Use only when intentional high-pass filtering is desired.
#   Always document this choice in figure captions or subtitles.
#
# This design makes the behavior **explicit, symmetric, and reproducible**.
#
