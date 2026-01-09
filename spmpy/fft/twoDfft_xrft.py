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

# %%
import numpy as np
import xarray as xr
import xrft


def twoDfft_xrft(
    ds: xr.Dataset,
    ch: str = "all",
    overwrite: bool = False,
    mask=None,
):
    """
    Perform a safe 2D Fourier transform on STM images using xrft.

    FFT is computed using xrft.fft to preserve coordinate consistency
    and physical frequency axes.

    NetCDF-safe complex storage
    ---------------------------
    - Complex FFT results are internally computed.
    - For NetCDF compatibility, complex values are additionally stored as:
        * <var>_fft_complex_real
        * <var>_fft_complex_imag
    - Amplitude and phase are always stored.
    """

    if not isinstance(ds, xr.Dataset):
        raise TypeError("Input must be an xarray.Dataset")

    if ch == "all":
        ch_list = list(ds.data_vars)
    else:
        if ch not in ds.data_vars:
            raise ValueError(f"Channel '{ch}' not found in Dataset")
        ch_list = [ch]

    out = ds.copy()

    for var in ch_list:
        da = out[var]
        if da.ndim != 2:
            continue

        data = da

        # ============================================================
        # Optional real-space mask (applied BEFORE FFT)
        # ============================================================
        if mask is not None:
            m = mask if isinstance(mask, xr.DataArray) else xr.DataArray(
                mask, dims=da.dims, coords=da.coords
            )
            data = data.where(m, 0.0)

        # ============================================================
        # FFT via xrft (NO manual fftshift / fftfreq)
        # ============================================================
        fft_da = xrft.fft(
            data,
            dim=("Y", "X"),
            true_phase=True,
            true_amplitude=True,
        )

        fft_complex = fft_da.values
        fft_amp = np.abs(fft_complex)
        fft_phase = np.angle(fft_complex)

        # ------------------------------------------------------------
        # Store complex FFT (in-memory)
        # ------------------------------------------------------------
        out[f"{var}_fft_complex"] = xr.DataArray(
            fft_complex,
            dims=fft_da.dims,
            coords=fft_da.coords,
            attrs=da.attrs,
        )

        # ------------------------------------------------------------
        # NetCDF-safe complex split
        # ------------------------------------------------------------
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
            fft_amp,
            dims=fft_da.dims,
            coords=fft_da.coords,
            attrs=da.attrs,
        )

        out[f"{var}_fft_phase"] = xr.DataArray(
            fft_phase,
            dims=fft_da.dims,
            coords=fft_da.coords,
            attrs=da.attrs,
        )

    # Automatic reciprocal reference (NetCDF-safe: scalar only)
    if "ref_a0_nm" in out.attrs:
        try:
            a0_nm = float(out.attrs["ref_a0_nm"])
            a0_m = a0_nm * 1e-9
            out.attrs["ref_q0_1overm"] = float(2.0 * np.pi / a0_m)
        except Exception:
            pass

    return out



# %%
import numpy as np
import xarray as xr
import xrft


def twoDifft_xrft(
    # [ADDED] Optional reciprocal-space mask support
    # mask: boolean array or xarray.DataArray, True = include in iFFT

    ds_fft: xr.Dataset,
    ch: str,
    use_complex: bool = True,
    overwrite: bool = False,
    mask=None,  # [ADDED] Optional FFT-space mask

):
    """
    Perform a safe inverse 2D Fourier transform to reconstruct real-space STM images
    using xrft.ifft.

    This function reconstructs real-space data from FFT-domain variables produced
    by `twoDfft_xrft`.

    IFFT input priority (NetCDF-safe)
    ---------------------------------
    1. If <ch>_fft_complex_real AND <ch>_fft_complex_imag exist:
        → reconstruct complex spectrum from real + imaginary parts
    2. Else if use_complex=True and <ch>_fft_complex exists:
        → use in-memory complex spectrum
    3. Else:
        → reconstruct complex spectrum from amplitude + phase

    NetCDF compatibility
    --------------------
    - Complex FFT data are split into real/imaginary components to ensure NetCDF safety.
    - This function transparently reconstructs the complex spectrum as needed.

    Mask behavior (additive, optional)
    ----------------------------------
    - If mask is None (default):
        Identical behavior to the original implementation.
    - If mask is provided:
        mask=True   → included in iFFT
        mask=False  → excluded (set to zero in frequency space)
    """

    if not isinstance(ds_fft, xr.Dataset):
        raise TypeError("Input must be an xarray.Dataset")

    if "freq_X" not in ds_fft.coords or "freq_Y" not in ds_fft.coords:
        raise ValueError("Dataset must contain 'freq_X' and 'freq_Y' coordinates")

    var = ch

    # ============================================================
    # Reconstruct complex FFT spectrum (priority order)
    # ============================================================
    if (
        f"{var}_fft_complex_real" in ds_fft
        and f"{var}_fft_complex_imag" in ds_fft
    ):
        fft_complex = (
            ds_fft[f"{var}_fft_complex_real"].values
            + 1j * ds_fft[f"{var}_fft_complex_imag"].values
        )

    elif use_complex and f"{var}_fft_complex" in ds_fft:
        fft_complex = ds_fft[f"{var}_fft_complex"].values

    else:
        if f"{var}_fft_amp" not in ds_fft or f"{var}_fft_phase" not in ds_fft:
            raise ValueError("Amplitude and phase required for reconstruction")
        amp = ds_fft[f"{var}_fft_amp"].values
        phase = ds_fft[f"{var}_fft_phase"].values
        fft_complex = amp * np.exp(1j * phase)

    # ============================================================
    # [ADDED] Apply reciprocal-space mask BEFORE iFFT
    # ============================================================
    if mask is not None:
        m = mask.values if hasattr(mask, "values") else mask
        fft_complex = np.where(m, fft_complex, 0.0)

    # ============================================================
    # Inverse FFT via xrft (coordinates preserved)
    # ============================================================
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

    # xrft.ifft returns complex; STM real-space signal is real-valued
    data_real = np.real(da_ifft.values)

    if "X" not in ds_fft.coords or "Y" not in ds_fft.coords:
        raise ValueError("Original real-space coordinates X/Y not found")

    out = ds_fft.copy()

    da_out = xr.DataArray(
        data_real,
        dims=("Y", "X"),
        coords={"Y": ds_fft["Y"], "X": ds_fft["X"]},
        attrs=ds_fft[ch].attrs if ch in ds_fft else {},
    )

    if overwrite:
        out[ch] = da_out
    else:
        out[f"{ch}_ifft"] = da_out

    return out


# %% [markdown]
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


# %% [markdown]
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

# %%
