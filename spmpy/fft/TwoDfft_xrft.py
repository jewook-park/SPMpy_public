# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

#
# # Patched TwoDfft_xrft (2026-01-02)
#
# This notebook provides a **fixed implementation of `TwoDfft_xrft`**.
#
# ### Fix summary
# - `np.angle()` now returns an **xarray.DataArray** (no `.attrs` crash)
# - FFT amplitude / phase / complex outputs are always valid DataArrays
# - Safe to use with SXM summary + PPTX pipeline
#

# +

import numpy as np
import xarray as xr
import xrft

# -

def TwoDfft_xrft(
    xrdata,
    ch='all',
    dims=('Y', 'X'),
    detrend='constant',
    window='hann',
    shift=True,
    overwrite=False,
):
    """
    Perform a safe 2D Fourier transform on xarray data using xrft.

    This function applies a 2D FFT to one or more channels in an
    xarray.Dataset or xarray.DataArray while **preserving coordinates,
    dimensions, and metadata**.

    Unlike naive numpy-based FFT workflows, this implementation ensures
    that all FFT-derived outputs (amplitude, phase, and complex spectrum)
    remain valid xarray.DataArray objects, making them safe for downstream
    analysis, visualization, and inverse FFT pipelines.

    Parameters
    ----------
    xrdata : xarray.Dataset or xarray.DataArray
        Input data containing 2D real-space images.
        Typical examples include:
            - Z_fwd, Z_bwd (topography)
            - LIX_fwd, LIX_bwd (lock-in signal)

    ch : str, default 'all'
        Channel to transform.
        - 'all' : apply FFT to all data variables in the Dataset
        - str   : apply FFT only to the specified channel

    dims : tuple of str, default ('Y', 'X')
        Names of the spatial dimensions over which the FFT is applied.
        These must exactly match the dimension names in the DataArray
        (case-sensitive).

        Example:
            dims=('Y','X')  # standard SXM images
            dims=('row','col')  # alternative naming

    detrend : {'constant', 'linear', None}, default 'constant'
        Detrending option passed to xrft.fft.
        - 'constant' : subtract mean value before FFT
        - 'linear'   : subtract best-fit plane
        - None       : no detrending

    window : {'hann', 'hamming', None}, default 'hann'
        Window function applied before FFT to reduce edge artifacts.

    shift : bool, default True
        If True, zero-frequency component is shifted to the center
        of Fourier space (fftshift behavior).

    overwrite : bool, default False
        Control how FFT results are stored:
        - False : original data are preserved and new variables are added
        - True  : original channel is replaced by FFT amplitude

    Returns
    -------
    xarray.Dataset
        Dataset containing FFT results.

        For each transformed channel ``var``, the following variables
        are added (unless overwrite=True):

        - ``var_fft_amp``     : FFT amplitude |F(k)|
        - ``var_fft_phase``   : FFT phase angle arg(F(k)) [radians]
        - ``var_fft_complex`` : Complex Fourier spectrum (xarray.DataArray)

    Notes
    -----
    - FFT amplitude and phase are stored separately for analysis and plotting.
    - The complex FFT result is retained to allow mathematically correct
      inverse FFT operations.
    - Phase information is essential; amplitude-only FFT data cannot
      reconstruct the original image.
    """

    # ------------------------------------------------------------
    # Normalize input to Dataset
    # ------------------------------------------------------------
    if isinstance(xrdata, xr.DataArray):
        ds = xrdata.to_dataset(name=xrdata.name or 'data')
    else:
        ds = xrdata.copy()

    # ------------------------------------------------------------
    # Determine which channels to FFT
    # ------------------------------------------------------------
    if ch == 'all':
        ch_list = list(ds.data_vars)
    else:
        if ch not in ds.data_vars:
            raise ValueError(f"Channel '{ch}' not found in Dataset")
        ch_list = [ch]

    # ------------------------------------------------------------
    # Apply FFT channel-by-channel
    # ------------------------------------------------------------
    for var in ch_list:
        da = ds[var]

        # Perform coordinate-aware FFT
        fft_da = xrft.fft(
            da,
            dim=dims,
            detrend=detrend,
            window=window,
            shift=shift,
        )

        # FFT amplitude as DataArray
        amp = xr.DataArray(
            np.abs(fft_da.values),
            coords=fft_da.coords,
            dims=fft_da.dims,
            attrs=da.attrs.copy(),
        )

        # FFT phase as DataArray (radians)
        phase = xr.DataArray(
            np.angle(fft_da.values),
            coords=fft_da.coords,
            dims=fft_da.dims,
            attrs=da.attrs.copy(),
        )

        # Annotate FFT representations
        amp.attrs['fft_representation'] = 'amplitude'
        phase.attrs['fft_representation'] = 'phase_rad'

        # Store results
        if overwrite:
            ds[var] = amp
        else:
            ds[f"{var}_fft_amp"] = amp
            ds[f"{var}_fft_phase"] = phase
            ds[f"{var}_fft_complex"] = fft_da

    return ds


#
# ## How to use
#
# ### Option A — Replace spmpy implementation
# Replace the contents of:
# ```
# spmpy/fft/TwoDfft_xrft.py
# ```
# with the function above.
#
# ### Option B — Temporary override in notebook
# Paste the function cell above directly into your summary notebook.
#
