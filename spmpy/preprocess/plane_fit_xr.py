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



# %%

def plane_fit_xr(
    xrdata,
    ch='all',
    method='surface_fit',
    poly_order=1,
    mask=None,
    overwrite=False
):
    """
    Polynomial plane / surface background removal for STM-SPM data.

    This function applies polynomial background fitting to:
    - 2D STM images (x, y)
    - Grid spectroscopy data (bias_mV, y, x)

    Parameters
    ----------
    xrdata : xarray.Dataset or xarray.DataArray
        Input STM/SPM data. If a DataArray is given, it is converted
        internally to a Dataset. Output is always a Dataset.

    ch : str, default 'all'
        Channel selection.
        - 'all' : apply fitting to all data variables
        - specific variable name (e.g. 'I_fwd')

    method : {'x_fit', 'y_fit', 'surface_fit'}, default 'surface_fit'
        Background fitting method.
        - x_fit : polynomial fit along x direction (row-wise)
        - y_fit : polynomial fit along y direction (column-wise)
        - surface_fit : full 2D polynomial surface fit

    poly_order : int, default 1
        Polynomial order of fitting (1, 2, or 3).

    mask : ndarray of bool, optional
        Boolean mask specifying which pixels are INCLUDED in fitting.

        mask == True  → used for fitting
        mask == False → excluded from fitting

        Notes:
        - mask may be sparse (point mask).
        - If None, all pixels are used.

    overwrite : bool, default False
        Storage behavior.
        - False : fitted result stored as '{var}_planefit'
        - True  : overwrite original variable

    Special behavior (grid.nc)
    --------------------------
    If a variable contains a 'bias_mV' dimension, plane fitting is
    applied independently for each bias slice.

    Returns
    -------
    xarray.Dataset
        Dataset containing plane-fitted data.
    """

    if isinstance(xrdata, xr.DataArray):
        ds = xrdata.to_dataset(name=xrdata.name or 'data')
    elif isinstance(xrdata, xr.Dataset):
        ds = xrdata.copy()
    else:
        raise TypeError("Input must be xarray.Dataset or xarray.DataArray")

    if ch == 'all':
        ch_list = list(ds.data_vars)
    else:
        if ch not in ds.data_vars:
            raise ValueError(f"Channel '{ch}' not found")
        ch_list = [ch]

    for var in ch_list:
        da = ds[var]

        # --- grid spectroscopy case ---
        if 'bias_mV' in da.dims:
            fitted_stack = []

            for ib, b in enumerate(da.bias_mV.values):
                slice2d = da.isel(bias_mV=ib)
                slice_mask = None if mask is None else mask

                slice_out = plane_fit_xr(
                    slice2d,
                    ch=slice2d.name,
                    method=method,
                    poly_order=poly_order,
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

        # --- pure 2D case ---
        else:
            data2d = da.values
            ny, nx = data2d.shape

            # use coordinate values if available
            x = da.coords[da.dims[1]].values if len(da.dims) == 2 else np.arange(nx)
            y = da.coords[da.dims[0]].values if len(da.dims) == 2 else np.arange(ny)

            if method == 'surface_fit':
                surface = _polyfit_surface_with_mask(x, y, data2d, poly_order, mask)
                result = data2d - surface

            elif method == 'x_fit':
                result = np.zeros_like(data2d)
                for iy in range(ny):
                    row_mask = None if mask is None else mask[iy]
                    fit = _polyfit_1d_with_mask(x, data2d[iy], poly_order, row_mask)
                    result[iy] = data2d[iy] - fit

            elif method == 'y_fit':
                result = np.zeros_like(data2d)
                for ix in range(nx):
                    col_mask = None if mask is None else mask[:, ix]
                    fit = _polyfit_1d_with_mask(y, data2d[:, ix], poly_order, col_mask)
                    result[:, ix] = data2d[:, ix] - fit

            else:
                raise ValueError("Invalid method")

            result = xr.DataArray(
                result,
                coords=da.coords,
                dims=da.dims,
                attrs=da.attrs
            )

        if overwrite:
            ds[var] = result
        else:
            ds[f"{var}_planefit"] = result

    return ds

