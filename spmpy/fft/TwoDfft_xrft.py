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
# # twoDfft_xrft — 2D FFT Utility for STM Images (xarray)
#
# This notebook documents **`twoDfft_xrft`**, a 2D Fourier-transform utility for
# STM (sxm-style) images stored in an `xarray.Dataset`.
#
# The function computes 2D FFTs for real-space channels (e.g. `Z_fwd`, `LIX_fwd`)
# and stores the results as:
#
# - `{var}_fft_complex`
# - `{var}_fft_amp`
# - `{var}_fft_phase`
#
# with corresponding reciprocal-space coordinates `freq_X`, `freq_Y` (in **1/m**).
#
# ---
#
# ## Automatic Reciprocal-Space Reference (`ref_q0_1overm`)
#
# If the input dataset contains the attribute:
#
# ```
# ds.attrs["ref_a0_nm"]
# ```
#
# then `twoDfft_xrft` will automatically add:
#
# ```
# ds_fft.attrs["ref_q0_1overm"] = 2π / a0
# ```
#
# where:
# - `a0` is converted from **nm → m**
# - `ref_q0_1overm` has units of **1/m**
#
# ---
#
# ## Log-scale Plotting of FFT Amplitude (Recommended)
#
# FFT amplitudes typically span many orders of magnitude.
# For visualization, it is strongly recommended to use **logarithmic color scaling**
# *at the plotting stage* (not by modifying the data).
#
# ### Example
#
# ```python
# from matplotlib.colors import LogNorm
#
# ds_sxm_fft.Z_fwd_fft_amp.plot(
#     norm=LogNorm()
# )
# ```
#
# This produces a true log-scale colorbar while preserving the original FFT data.
#
# ---
#
# ## Typical Usage
#
# ```python
# ds_sxm_fft = twoDfft_xrft(
#     ds_sxm,
#     ch="Z_fwd"
# )
# ```
#
# The resulting dataset will contain both real-space data and FFT-domain data,
# all stored in a single `xarray.Dataset`.
#

# %%

import numpy as np
import xarray as xr


def twoDfft_xrft(
    ds: xr.Dataset,
    ch: str = "all",
    overwrite: bool = False,
):
    """
    Perform a safe 2D Fourier transform on STM images using xarray-compatible FFT.
    
    This function computes a 2D Fourier transform for one or more real-space
    STM channels stored in an xarray.Dataset, while **preserving coordinates,
    dimensions, and metadata** throughout the FFT pipeline.
    
    Unlike naive NumPy-based FFT workflows, this implementation ensures that
    all FFT-derived outputs (amplitude, phase, and complex spectrum) remain
    valid xarray.DataArray objects. This makes the results safe for downstream
    analysis, visualization, and mathematically consistent inverse FFT
    operations.
    
    The function is designed for **sxm-style 2D STM images** and follows a
    well-defined variable-naming and metadata convention consistent with SPMpy.
    
    ----------------------------------------------------------------------
    FFT Outputs
    ----------------------------------------------------------------------
    For each transformed channel ``var``, the following variables are produced
    (unless overwrite=True):
    
    - ``var_fft_complex`` : complex Fourier spectrum F(kx, ky)
    - ``var_fft_amp``     : FFT amplitude |F(kx, ky)|
    - ``var_fft_phase``   : FFT phase arg(F(kx, ky)) [radians]
    
    Reciprocal-space coordinates are added as:
    
    - ``freq_X`` : spatial frequency along X (unit: 1/m)
    - ``freq_Y`` : spatial frequency along Y (unit: 1/m)
    
    ----------------------------------------------------------------------
    Automatic Reciprocal Reference
    ----------------------------------------------------------------------
    If the input dataset contains the attribute:
    
        ds.attrs["ref_a0_nm"]
    
    (the real-space lattice constant in nanometers),
    
    the function automatically adds the reciprocal-space reference:
    
        ds_fft.attrs["ref_q0_1overm"] = 2π / a0
    
    where ``a0`` is converted from nanometers to meters.  
    The resulting ``ref_q0_1overm`` has units of **1/m** and is useful for
    annotating FFT plots with physically meaningful reciprocal-lattice scales.
    
    ----------------------------------------------------------------------
    Visualization Notes (FFT Amplitude)
    ----------------------------------------------------------------------
    FFT amplitudes typically span many orders of magnitude.
    For visualization, logarithmic color scaling is strongly recommended
    **at the plotting stage**, not by modifying the data.
    
    Example:
    
        from matplotlib.colors import LogNorm
        ds_sxm_fft.Z_fwd_fft_amp.plot(norm=LogNorm())
    
    This produces a true log-scale colorbar while preserving the original
    FFT amplitude values.
    
    ----------------------------------------------------------------------
    Parameters
    ----------------------------------------------------------------------
    xrdata : xarray.Dataset or xarray.DataArray
        Input data containing 2D real-space STM images.
        Typical examples include:
            - Z_fwd, Z_bwd   (topography)
            - LIX_fwd, LIX_bwd (lock-in signals)
    
    ch : str, default 'all'
        Channel(s) to transform.
        - 'all' : apply FFT to all 2D data variables in the Dataset
        - str   : apply FFT only to the specified channel
    
    dims : tuple of str, default ('Y', 'X')
        Names of the spatial dimensions over which the FFT is applied.
        These must exactly match the DataArray dimension names
        (case-sensitive).
    
        Examples:
            dims=('Y','X')        # standard SXM images
            dims=('row','col')    # alternative naming
    
    detrend : {'constant', 'linear', None}, default 'constant'
        Detrending option applied before FFT.
        - 'constant' : subtract mean value
        - 'linear'   : subtract best-fit plane
        - None       : no detrending
    
    window : {'hann', 'hamming', None}, default 'hann'
        Window function applied before FFT to reduce edge artifacts.
    
    shift : bool, default True
        If True, shift the zero-frequency component to the center of
        Fourier space (fftshift behavior).
    
    overwrite : bool, default False
        Control how FFT results are stored:
        - False : original data are preserved and FFT variables are appended
        - True  : original channel is replaced by FFT amplitude
    
    ----------------------------------------------------------------------
    Returns
    ----------------------------------------------------------------------
    xarray.Dataset
        Dataset containing original real-space data and FFT-domain variables,
        including amplitude, phase, and complex Fourier spectra.
    
    ----------------------------------------------------------------------
    Notes
    ----------------------------------------------------------------------
    - FFT amplitude and phase are stored separately for clarity and analysis.
    - The complex FFT spectrum is retained to allow mathematically correct
      inverse FFT operations.
    - Phase information is essential; amplitude-only FFT data cannot
      reconstruct the original real-space image.
    """


    if not isinstance(ds, xr.Dataset):
        raise TypeError("Input must be an xarray.Dataset")

    # Channel list
    if ch == "all":
        ch_list = list(ds.data_vars)
    else:
        if ch not in ds.data_vars:
            raise ValueError(f"Channel '{ch}' not found in Dataset")
        ch_list = [ch]

    out = ds.copy()

    if "X" not in out.coords or "Y" not in out.coords:
        raise ValueError("Dataset must contain 'X' and 'Y' coordinates")

    X = out["X"].values
    Y = out["Y"].values

    dx = float(X[1] - X[0])
    dy = float(Y[1] - Y[0])

    Nx = len(X)
    Ny = len(Y)

    freq_X = np.fft.fftshift(np.fft.fftfreq(Nx, d=dx))
    freq_Y = np.fft.fftshift(np.fft.fftfreq(Ny, d=dy))

    out = out.assign_coords(
        freq_X=("freq_X", freq_X),
        freq_Y=("freq_Y", freq_Y),
    )

    for var in ch_list:
        da = out[var]
        if da.ndim != 2:
            continue

        data = da.values.astype(float)

        fft_complex = np.fft.fftshift(np.fft.fft2(data))
        fft_amp = np.abs(fft_complex)
        fft_phase = np.angle(fft_complex)

        out[f"{var}_fft_complex"] = xr.DataArray(
            fft_complex,
            dims=("freq_Y", "freq_X"),
            coords={"freq_Y": freq_Y, "freq_X": freq_X},
            attrs=da.attrs,
        )

        out[f"{var}_fft_amp"] = xr.DataArray(
            fft_amp,
            dims=("freq_Y", "freq_X"),
            coords={"freq_Y": freq_Y, "freq_X": freq_X},
            attrs=da.attrs,
        )

        out[f"{var}_fft_phase"] = xr.DataArray(
            fft_phase,
            dims=("freq_Y", "freq_X"),
            coords={"freq_Y": freq_Y, "freq_X": freq_X},
            attrs=da.attrs,
        )

    # Automatic reciprocal reference from ref_a0_nm
    if "ref_a0_nm" in out.attrs:
        try:
            a0_nm = float(out.attrs["ref_a0_nm"])
            a0_m = a0_nm * 1e-9
            out.attrs["ref_q0_1overm"] = float(2.0 * np.pi / a0_m)
        except Exception:
            pass

    return out

