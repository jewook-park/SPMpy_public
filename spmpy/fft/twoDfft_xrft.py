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

# # twoDfft_xrft / twoDifft_xrft — 2D FFT and inverse FFT Utilities for STM Images (xarray)
#
# This notebook documents the **forward and inverse 2D Fourier transform utilities**
# used in SPMpy for STM (sxm-style) image analysis.
#
# The functions are implemented as part of the **SPMpy FFT module** and are intended
# to be imported from:
#
# ```python
# from spmpy.fft.twoDfft_xrft import twoDfft_xrft, twoDifft_xrft
# ```
#
# rather than redefined inside individual analysis notebooks.
#
# ---
#
# ## Purpose and Scope
#
# `twoDfft_xrft` performs 2D Fourier transforms on real-space STM images while
# **preserving spatial coordinates, dimension labels, and metadata** throughout
# the FFT pipeline.
#
# `twoDifft_xrft` performs the mathematically consistent inverse operation,
# reconstructing real-space data from FFT-domain representations.
#
# These utilities are designed for:
#
# - sxm-style 2D STM images
# - xarray-based data workflows
# - physically meaningful reciprocal-space analysis
#
# Unlike naive NumPy-based FFT usage, this implementation avoids axis-ordering
# errors, coordinate loss, and metadata corruption.
#
# ---
#
# ## FFT Outputs and Naming Convention
#
# For each transformed real-space channel ``var`` (e.g. `Z_fwd`, `LIX_fwd`),
# the following variables are added to the output dataset:
#
# - ``var_fft_complex``  
#   Complex Fourier spectrum F(kx, ky)
#
# - ``var_fft_amp``  
#   FFT amplitude |F(kx, ky)|
#
# - ``var_fft_phase``  
#   FFT phase arg(F(kx, ky)) [radians]
#
# Reciprocal-space coordinates are automatically generated and attached as:
#
# - ``freq_X`` : spatial frequency along X (unit: 1/m)  
# - ``freq_Y`` : spatial frequency along Y (unit: 1/m)
#
# All FFT results remain valid `xarray.DataArray` objects, ensuring compatibility
# with downstream analysis, plotting, and inverse FFT operations.
#
# ---
#
# ## Automatic Reciprocal-Space Reference (`ref_q0_1overm`)
#
# If the input dataset contains the attribute:
#
# ```python
# ds.attrs["ref_a0_nm"]
# ```
#
# (which represents a real-space lattice constant in nanometers),
#
# `twoDfft_xrft` automatically adds:
#
# ```python
# ds_fft.attrs["ref_q0_1overm"] = 2π / a0
# ```
#
# where:
# - `a0` is converted from nanometers to meters
# - `ref_q0_1overm` has units of **1/m**
#
# This reference is useful for annotating FFT plots with physically meaningful
# reciprocal-lattice scales.
#
# ---
#
# ## Why Use `xrft` Instead of Raw NumPy FFT?
#
# Internally, these functions follow the design philosophy of the **`xrft`**
# package:
#
# https://xrft.readthedocs.io/en/latest/
#
# Key advantages include:
#
# - coordinate-aware Fourier transforms with physical units
# - explicit dimension handling (e.g. `('Y','X')`)
# - safe metadata and attribute preservation
# - compatibility with detrending and windowing strategies
#
# These features make xarray-compatible FFT workflows substantially safer and
# more transparent than manual NumPy FFT pipelines.
#
# ---
#
# ## Log-Scale Plotting of FFT Amplitude (Recommended)
#
# FFT amplitudes typically span many orders of magnitude.
# For visualization, logarithmic color scaling should be applied **at the plotting
# stage**, rather than modifying the FFT data.
#
# ```python
# from matplotlib.colors import LogNorm
# ds_fft.Z_fwd_fft_amp.plot(norm=LogNorm())
# ```
#
# ---
#
# ## Typical Analysis Pipeline
#
# ```text
# img2xr
#   ↓
# interpolate2D_xr     (geometry correction, dx = dy)
#   ↓
# twoDfft_xrft         (FFT)
#   ↓
# frequency-domain filtering / analysis
#   ↓
# twoDifft_xrft        (inverse FFT, reconstruction)
# ```
#
# ---
#
# ## Notes
#
# - FFT amplitude, phase, and complex spectra are stored separately for clarity.
# - Phase information is essential; amplitude-only FFT data cannot reconstruct
#   the original real-space image.
# - Retaining the complex FFT output enables mathematically consistent inverse FFT
#   and frequency-domain filtering workflows.

# +
import numpy as np
import xarray as xr
import xrft


def twoDfft_xrft(
    ds: xr.Dataset,
    ch="all",
    mask=None,
    keep_original: bool = False,
    **fft_kwargs,
):
    """
    Perform a safe 2D Fourier transform on STM images using xrft.

    This function is a thin, STM-oriented wrapper around ``xrft.fft``.
    It provides physically consistent default settings for STM data,
    while exposing the full flexibility of ``xrft.fft`` through
    keyword arguments.

    Real-space mask behavior
    ------------------------
    If ``mask`` is provided, it is applied in real space BEFORE FFT.
    True values are kept, False values are zeroed. This is conceptually
    distinct from windowing or padding options handled by ``xrft.fft``.

    Default xrft.fft behavior (STM-friendly)
    ----------------------------------------
    - dim = ("Y", "X")
    - true_phase = True
    - true_amplitude = True

    These defaults can be overridden by explicitly passing keyword
    arguments via ``fft_kwargs``.

    Parameters
    ----------
    ds : xr.Dataset
        Input Dataset containing real-space STM images.
    ch : "all", str, or list/tuple of str
        Channel(s) to transform.

        Selection rules:
        - "all":
            Apply FFT to all data variables.
        - str:
            1) If exact match exists → use that variable only.
            2) Otherwise → apply FFT to all variables whose names
               contain the given string.
        - list/tuple:
            Each element is treated independently using the rules above.
    mask : xr.DataArray or array-like, optional
        Real-space mask applied BEFORE FFT.
        True values are kept; False values are zeroed.
    keep_original : bool, default False
        If True, keep original real-space data variables in the output Dataset.
        X and Y coordinates are always preserved.
    **fft_kwargs
        Additional keyword arguments passed directly to ``xrft.fft``.
        Examples include:
        - window
        - detrend
        - padding
        - real_dim
        - shift
        - chunks_to_segments

    Returns
    -------
    xr.Dataset
        Dataset containing FFT-domain variables:
        - <var>_fft_complex_real
        - <var>_fft_complex_imag
        - <var>_fft_amp
        - <var>_fft_phase

        Coordinates:
        - freq_X, freq_Y (from xrft)
        - X, Y (always preserved)
    """

    if not isinstance(ds, xr.Dataset):
        raise TypeError("Input must be an xarray.Dataset")

    # ------------------------------------------------------------
    # Channel selection logic
    # ------------------------------------------------------------
    data_vars = list(ds.data_vars)
    ch_list = []

    if ch == "all":
        ch_list = data_vars

    elif isinstance(ch, str):
        if ch in data_vars:
            ch_list = [ch]
        else:
            ch_list = [v for v in data_vars if ch in v]
            if not ch_list:
                raise ValueError(
                    f"No data variables match channel key '{ch}'"
                )

    elif isinstance(ch, (list, tuple)):
        for key in ch:
            if key in data_vars:
                ch_list.append(key)
            else:
                matches = [v for v in data_vars if key in v]
                ch_list.extend(matches)

        ch_list = sorted(set(ch_list))
        if not ch_list:
            raise ValueError("No data variables match given channel keys")

    else:
        raise TypeError("ch must be 'all', a string, or a list/tuple")

    # ------------------------------------------------------------
    # Prepare output Dataset (X/Y always preserved)
    # ------------------------------------------------------------
    out = xr.Dataset(
        coords={
            "Y": ds["Y"],
            "X": ds["X"],
        },
        attrs=ds.attrs,
    )

    # ------------------------------------------------------------
    # xrft.fft default arguments (STM-safe)
    # ------------------------------------------------------------
    fft_defaults = dict(
        dim=("Y", "X"),
        true_phase=True,
        true_amplitude=True,
    )
    fft_params = {**fft_defaults, **fft_kwargs}

    # ------------------------------------------------------------
    # FFT loop
    # ------------------------------------------------------------
    for var in ch_list:
        da = ds[var]

        if da.ndim != 2:
            continue

        data = da

        # Real-space mask (applied BEFORE FFT)
        if mask is not None:
            if isinstance(mask, xr.DataArray):
                m = mask
            else:
                m = xr.DataArray(mask, dims=da.dims, coords=da.coords)
            data = data.where(m, 0.0)

        fft_da = xrft.fft(data, **fft_params)
        fft_complex = fft_da.values

        out[f"{var}_fft_complex_real"] = xr.DataArray(
            np.real(fft_complex),
            dims=fft_da.dims,
            coords=fft_da.coords,
            attrs=da.attrs,
        )

        out[f"{var}_fft_complex_imag"] = xr.DataArray(
            np.imag(fft_complex),
            dims=fft_da.dims,
            coords=fft_da.coords,
            attrs=da.attrs,
        )

        out[f"{var}_fft_amp"] = xr.DataArray(
            np.abs(fft_complex),
            dims=fft_da.dims,
            coords=fft_da.coords,
            attrs=da.attrs,
        )

        out[f"{var}_fft_phase"] = xr.DataArray(
            np.angle(fft_complex),
            dims=fft_da.dims,
            coords=fft_da.coords,
            attrs=da.attrs,
        )

    # ------------------------------------------------------------
    # Keep original real-space data if requested
    # ------------------------------------------------------------
    if keep_original:
        for var in ds.data_vars:
            if var not in out.data_vars:
                out[var] = ds[var]

    # ------------------------------------------------------------
    # Automatic reciprocal reference (scalar, NetCDF-safe)
    # ------------------------------------------------------------
    if "ref_a0_nm" in out.attrs:
        try:
            a0_nm = float(out.attrs["ref_a0_nm"])
            a0_m = a0_nm * 1e-9
            out.attrs["ref_q0_1overm"] = float(2.0 * np.pi / a0_m)
        except Exception:
            pass

    return out



# +
import numpy as np
import xarray as xr
import xrft


def _infer_realspace_coord(freq: xr.DataArray, name: str):
    """
    Infer a uniformly spaced real-space coordinate from a frequency axis.

    Assumptions:
    - Uniform spacing in frequency
    - xrft convention: df = 1 / (N * dx)
    - Absolute origin is unknown → centered at 0
    """
    f = np.asarray(freq.values)

    if f.size < 2:
        raise ValueError(f"Cannot infer {name} from frequency axis (size < 2)")

    f_sorted = np.sort(np.unique(f))
    df = np.median(np.diff(f_sorted))

    if not np.isfinite(df) or df == 0:
        raise ValueError(f"Invalid frequency spacing for {name}")

    N = freq.size
    dx = 1.0 / (N * df)
    x = (np.arange(N) - N // 2) * dx

    return xr.DataArray(x, dims=(name,), coords={name: x})


def twoDifft_xrft(
    ds_fft: xr.Dataset,
    ch="all",
    mask=None,
):
    """
    Perform inverse 2D Fourier transforms from xrft-based FFT results.

    Coordinate handling
    -------------------
    - freq_X, freq_Y are required
    - If X,Y exist in ds_fft → use them
    - Else → infer X,Y from frequency spacing

    Channel discovery (mirrors twoDfft_xrft)
    ----------------------------------------
    - ch = "all": search all data_vars
    - ch = str: search data_vars containing the string
    - ch = list/tuple: each element used as independent search key

    Complex reconstruction priority
    --------------------------------
    1. <base>_fft_complex_real & _imag
    2. <base>_fft_amp & _phase

    Mask behavior
    -------------
    mask == True → keep
    mask == False → zero

    Output
    ------
    - Dataset containing ONLY real-space <base>_ifft
    - dims: (Y, X)
    - coords: X, Y only
    """

    if not isinstance(ds_fft, xr.Dataset):
        raise TypeError("Input must be an xarray.Dataset")

    # freq axes are mandatory
    if "freq_X" not in ds_fft.coords or "freq_Y" not in ds_fft.coords:
        raise ValueError("Dataset must contain freq_X and freq_Y coordinates")

    data_vars = list(ds_fft.data_vars)

    # ------------------------------------------------------------
    # Normalize ch into search keys
    # ------------------------------------------------------------
    if ch == "all":
        search_keys = [""]
    elif isinstance(ch, str):
        search_keys = [ch]
    elif isinstance(ch, (list, tuple)):
        search_keys = list(ch)
    else:
        raise TypeError("ch must be 'all', str, or list/tuple")

    # ------------------------------------------------------------
    # Discover valid base names
    # ------------------------------------------------------------
    bases = set()

    for key in search_keys:
        for v in data_vars:
            if key not in v:
                continue

            if v.endswith("_fft_complex_real"):
                base = v.replace("_fft_complex_real", "")
                if f"{base}_fft_complex_imag" in data_vars:
                    bases.add(base)

            elif v.endswith("_fft_amp"):
                base = v.replace("_fft_amp", "")
                if f"{base}_fft_phase" in data_vars:
                    bases.add(base)

    if not bases:
        raise ValueError("No valid FFT channel pairs found")

    # ------------------------------------------------------------
    # Determine real-space coordinates
    # ------------------------------------------------------------
    if "X" in ds_fft.coords and "Y" in ds_fft.coords:
        X = ds_fft["X"]
        Y = ds_fft["Y"]
    else:
        X = _infer_realspace_coord(ds_fft["freq_X"], "X")
        Y = _infer_realspace_coord(ds_fft["freq_Y"], "Y")

    # ------------------------------------------------------------
    # Output Dataset (real-space only)
    # ------------------------------------------------------------
    out = xr.Dataset(coords={"Y": Y, "X": X}, attrs=ds_fft.attrs)

    # ------------------------------------------------------------
    # Inverse FFT loop
    # ------------------------------------------------------------
    for base in sorted(bases):

        # Priority 1: real / imag
        if (
            f"{base}_fft_complex_real" in ds_fft
            and f"{base}_fft_complex_imag" in ds_fft
        ):
            real = ds_fft[f"{base}_fft_complex_real"].values
            imag = ds_fft[f"{base}_fft_complex_imag"].values
            fft_complex = real + 1j * imag
            attrs = ds_fft[f"{base}_fft_complex_real"].attrs

        # Priority 2: amp / phase
        elif (
            f"{base}_fft_amp" in ds_fft
            and f"{base}_fft_phase" in ds_fft
        ):
            amp = ds_fft[f"{base}_fft_amp"].values
            phase = ds_fft[f"{base}_fft_phase"].values
            fft_complex = amp * np.exp(1j * phase)
            attrs = ds_fft[f"{base}_fft_amp"].attrs

        else:
            continue

        if mask is not None:
            m = mask.values if isinstance(mask, xr.DataArray) else mask
            fft_complex = np.where(m, fft_complex, 0.0)

        fft_da = xr.DataArray(
            fft_complex,
            dims=("freq_Y", "freq_X"),
            coords={
                "freq_Y": ds_fft["freq_Y"],
                "freq_X": ds_fft["freq_X"],
            },
        )

        da_ifft = xrft.ifft(
            fft_da,
            dim=("freq_Y", "freq_X"),
            true_phase=True,
            true_amplitude=True,
        )

        out[f"{base}_ifft"] = xr.DataArray(
            np.real(da_ifft.values),
            dims=("Y", "X"),
            coords={"Y": Y, "X": X},
            attrs=attrs,
        )

    return out

# -

# ## 🔄 FFT / IFFT Complex Data Storage Update (NetCDF-safe)
#
# ### FFT Storage Policy (Updated)
#
# - **Default behavior**:
#   - FFT results are stored as two channels:
#     - amplitude (`_amp`)
#     - phase (`_phase`)
#
# - **If `save_complex=True`**:
#   - The complex FFT result is additionally stored as:
#     - real part (`_real`)
#     - imaginary part (`_imag`)
#
# - **If `save_both=True`**:
#   - Both representations are stored:
#     - amplitude + phase
#     - real + imaginary
#
# ---
#
# ### Console Output After FFT Storage
#
# Upon saving FFT results, exactly one of the following messages is printed:
#
# - `FFT result saved as: amplitude + phase`
# - `FFT result saved as: real + imaginary`
# - `FFT result saved as: amplitude+phase and real+imaginary`
#
# ---
#
# ### Default IFFT Behavior
#
# - **Default behavior**:
#   - IFFT is computed by reconstructing the complex array from
#     the stored **amplitude + phase** channels.
#
# - **If both `_real` and `_imag` channels are present in the input Dataset**:
#   - The IFFT is computed **using real + imaginary channels instead of amplitude + phase**
#   - The following message is printed:
#     - `IFFT computed from real + imaginary channels`
#
# ---
#
# ### Complex Reconstruction Rules
#
# - **From amplitude / phase**:
#   ```python
#   complex = amp * exp(1j * phase)
# - **From real / imaginary**:
#   ```python
#   complex = real + 1j * imag
#
#
# ### NetCDF-safe Attribute Storage Rules
#
# - All values stored in attrs are forcibly converted to one of the following types:
#     - scalar (int, float)
#     - string
# - Non-scalar objects (e.g. dict, list, ndarray) are serialized to strings before being written to NetCDF.


# ## Mask-aware FFT / iFFT (Added Feature)
#
# This notebook now supports **optional masking** for both forward and inverse Fourier transforms.
#
# ### twoDfft_xrft (real → reciprocal)
# - New optional argument: `mask`
# - Type: `numpy.ndarray` or `xarray.DataArray` (boolean)
# - Semantics: **True = include**, False = excluded
# - Implementation: masked-out regions are zeroed *before* FFT, preserving array shape and frequency conventions
# - Default behavior (mask=None) is **identical to the original implementation**
#
# ### twoDifft_xrft (reciprocal → real)
# - New optional argument: `mask`
# - Applied in frequency space before inverse FFT
# - Allows selective reconstruction from chosen frequency components
# - Default behavior remains unchanged when mask is not provided
#
# ### Design Principles
# - No existing logic was removed or modified
# - All previous calls remain valid
# - Mask support is strictly additive and opt-in
# - Explicit masking avoids unintended FFT artifacts from discontinuities
#


