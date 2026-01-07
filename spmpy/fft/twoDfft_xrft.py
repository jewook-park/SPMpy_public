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


def twoDfft_xrft(
    ds: xr.Dataset,
    ch: str = "all",
    overwrite: bool = False,
):
    """
    Perform a safe 2D Fourier transform on STM images using xarray-compatible FFT.

    This function computes a 2D Fourier transform for one or more real-space
    STM channels stored in an xarray.Dataset, while preserving coordinates,
    dimensions, and metadata throughout the FFT pipeline.

    FFT-domain outputs are stored as xarray.DataArray objects, ensuring safe
    downstream analysis and inverse FFT operations.
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


def twoDifft_xrft(
    ds_fft: xr.Dataset,
    ch: str,
    use_complex: bool = True,
    overwrite: bool = False,
):
    """
    Perform a safe inverse 2D Fourier transform to reconstruct real-space STM images.

    This function reconstructs real-space data from FFT-domain variables produced
    by `twoDfft_xrft`.

    By default, the inverse FFT uses the complex Fourier spectrum. If complex data
    are not available, the spectrum is reconstructed from amplitude and phase.

    All operations preserve xarray coordinates, dimensions, and metadata.
    """

    if not isinstance(ds_fft, xr.Dataset):
        raise TypeError("Input must be an xarray.Dataset")

    if "freq_X" not in ds_fft.coords or "freq_Y" not in ds_fft.coords:
        raise ValueError("Dataset must contain 'freq_X' and 'freq_Y' coordinates")

    var = ch

    if use_complex and f"{var}_fft_complex" in ds_fft:
        fft_complex = ds_fft[f"{var}_fft_complex"].values
    else:
        if f"{var}_fft_amp" not in ds_fft or f"{var}_fft_phase" not in ds_fft:
            raise ValueError("Amplitude and phase required for reconstruction")
        amp = ds_fft[f"{var}_fft_amp"].values
        phase = ds_fft[f"{var}_fft_phase"].values
        fft_complex = amp * np.exp(1j * phase)

    data_ifft = np.fft.ifft2(np.fft.ifftshift(fft_complex))
    data_real = np.real(data_ifft)

    if "X" not in ds_fft.coords or "Y" not in ds_fft.coords:
        raise ValueError("Original real-space coordinates X/Y not found")

    out = ds_fft.copy()

    da_ifft = xr.DataArray(
        data_real,
        dims=("Y", "X"),
        coords={"Y": ds_fft["Y"], "X": ds_fft["X"]},
        attrs=ds_fft[ch].attrs if ch in ds_fft else {},
    )

    if overwrite:
        out[ch] = da_ifft
    else:
        out[f"{ch}_ifft"] = da_ifft

    return out


# %% [markdown]
#
# ## 🔄 FFT / IFFT Complex Data Storage Update (NetCDF-safe)
#
# ### FFT 저장 규칙 (업데이트)
# - 기본(default):
#   - FFT 결과는 **amplitude (`_amp`)** 와 **phase (`_phase`)** 두 채널로 저장
# - 옵션 `save_complex=True` 인 경우:
#   - 복소 FFT 결과를 **real (`_real`)**, **imaginary (`_imag`)** 채널로 추가 저장
# - 옵션 `save_both=True` 인 경우:
#   - `amp/phase` 와 `real/imag` **모두 저장**
#
# ### FFT 저장 후 출력 메시지
# - 저장 시 다음 중 하나를 명시적으로 출력:
#   - `FFT result saved as: amplitude + phase`
#   - `FFT result saved as: real + imaginary`
#   - `FFT result saved as: amplitude+phase and real+imaginary`
#
# ### IFFT 기본 동작
# - 기본(default):
#   - 저장된 **amplitude + phase** 로부터 복소수를 재구성하여 IFFT 수행
# - 만약 입력 Dataset에:
#   - `_real` 과 `_imag` 가 **모두 존재**하는 경우:
#     - `_amp`, `_phase` 대신 **real + imaginary 기반**으로 IFFT 수행
#     - 출력 메시지:
#       - `IFFT computed from real + imaginary channels`
#
# ### 복소수 재구성 규칙
# - amplitude / phase 기반:
#   - `complex = amp * exp(1j * phase)`
# - real / imaginary 기반:
#   - `complex = real + 1j * imag`
#
# ### NetCDF-safe attrs 저장 규칙
# - 모든 attrs 값은 다음 중 하나로 강제 변환:
#   - scalar (int, float)
#   - string
# - dict, list, ndarray 등은 문자열로 변환하여 저장
#

# %%

import numpy as np
import xarray as xr

def fft2d_save(ds, var, save_complex=False, save_both=False):
    data = ds[var]
    fft_complex = np.fft.fftshift(np.fft.fft2(data))

    amp = np.abs(fft_complex)
    phase = np.angle(fft_complex)

    out = xr.Dataset()

    out[f"{var}_amp"] = (data.dims, amp)
    out[f"{var}_phase"] = (data.dims, phase)

    msg = "FFT result saved as: amplitude + phase"

    if save_complex or save_both:
        out[f"{var}_real"] = (data.dims, fft_complex.real)
        out[f"{var}_imag"] = (data.dims, fft_complex.imag)
        msg = "FFT result saved as: real + imaginary"

    if save_both:
        msg = "FFT result saved as: amplitude+phase and real+imaginary"

    print(msg)

    # attrs sanitize
    out.attrs = {
        k: (float(v) if np.isscalar(v) else str(v))
        for k, v in ds.attrs.items()
    }

    return out


def ifft2d_from_ds(ds, var):
    if f"{var}_real" in ds and f"{var}_imag" in ds:
        complex_data = ds[f"{var}_real"] + 1j * ds[f"{var}_imag"]
        print("IFFT computed from real + imaginary channels")
    else:
        amp = ds[f"{var}_amp"]
        phase = ds[f"{var}_phase"]
        complex_data = amp * np.exp(1j * phase)
        print("IFFT computed from amplitude + phase channels")

    ifft_data = np.fft.ifft2(np.fft.ifftshift(complex_data))

    return xr.DataArray(
        np.real(ifft_data),
        dims=complex_data.dims,
        coords=complex_data.coords,
        name=f"{var}_ifft"
    )

