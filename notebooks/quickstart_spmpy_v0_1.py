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
# # SPMpy Quickstart v0.1
#
# SPMpy is an open-source collection of Python tools for analyzing multi-dimensional scanning probe microscopy (SPM) data,
# including STM/S and AFM. It uses **`xarray`** as the primary data container to preserve both data and metadata.
#
# **Authors:** Dr. Jewook Park (CNMS, ORNL)  
# **Contact:** parkj1@ornl.gov
#
# ### Stages in this notebook
# - **Stage 0:** Environment check + repository bootstrap
# - **Stage 1:** Data loading (Nanonis `.sxm`, `.3ds`) into `xarray.Dataset`
# - **Stage 2:** (Basic) Visualization and analysis 
#
# ### License note
# This repository is provided for internal and collaborative review. Licensing terms will be finalized according to ORNL/DOE policies.
#

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# <a id="Navigation"></a>
#
# ## Notebook Navigation
#
# - [**Stage 0**](#Stage0) — Environment check & bootstrap
#   - Step 1: Set `REPO_ROOT` and import `spmpy`
#   - Step 2: Run structured environment diagnostics
#   - Step 3: Environment diagnostics 
# - [**Stage 1**](#Stage1) — Data loading (STM/SPM files)
#   - [Stage 1.1](#Stage1.1): 2D image data (`.sxm`) → `xarray.Dataset`
#   - [Stage 1.2](#Stage1.2): GridSpectroscopy (`.3ds`) → `xarray.Dataset`
# - [**Stage 2**](#Stage2) — Data-processing & visualization 
#     - [Stage2.1] (#`ds_sxm` preprocessing)
#
# **Tip:** Run cells from top to bottom. Markdown cells describe what to do and what to expect.
#

# %% [markdown]
# <a id="Stage0"></a>
#
# # Stage 0 — Step 1: Bootstrap the local repository
#
# [⬆ Back to Navigation](#Navigation)
#
#
# Set `REPO_ROOT` to your local SPMpy clone folder, add it to `sys.path`, then import `spmpy`.
#
#
# ### Installation and Loading (Internal Use)
#
# Since this repository is now **private**, SPMpy is not installed via public package managers
# (e.g., `pip install`) and should be accessed **only through direct cloning** of the repository.
#
# #### Access Requirement
# To use SPMpy, you must:
# - Have received an **invitation** to this private GitHub repository
#
# If you do not have access, please contact the repository maintainer, Jewook Park (parkj1@ornl.gov).
#
# ---
#
# #### Clone the Private Repository
#
# After accepting the GitHub invitation, clone the repository to your local machine:
#
# ```bash
# git clone git@github.com:jewook-park/SPMpy.git

# %%
import sys
from pathlib import Path

# IMPORTANT: set this to your local SPMpy repository root
REPO_ROOT = Path(r"C:\\Users\\gkp\\Documents\\GitHub\\SPMpy")

if not REPO_ROOT.exists():
    raise RuntimeError(
        f"[SPMpy] Repo root does not exist: {REPO_ROOT}\n"
        "[Action] Edit REPO_ROOT to match your local clone location."
    )

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import spmpy
print('[SPMpy] Imported from:', spmpy.__file__)


# %% [markdown]
# ## Stage-0 Step 2 — Environment diagnostic (read-only)

# %%
# Safe import of env_check module (explicit module path)
try:
    import spmpy.utils.env_check_v2025Dec_30_revised as env_check
except ImportError as e:
    raise RuntimeError(
        '[SPMpy] Failed to import env_check module.\n'
        'Reason: module file not found or misnamed.\n'
        'Action: verify file name and restart kernel.'
    ) from e


# %%
from dataclasses import dataclass

@dataclass
class EnvStatus:
    ok: bool = False
    needs_restart: bool = False
    inconclusive: bool = False
    missing: list | None = None

def interpret_env_check(env):
    status = EnvStatus()

    if hasattr(env, 'ENV_OK'):
        status.ok = bool(env.ENV_OK)
        status.needs_restart = bool(getattr(env, 'INSTALLED_NOW', False))
        status.missing = getattr(env, 'MISSING_REQUIRED', None)
        return status

    status.inconclusive = True
    return status

status = interpret_env_check(env_check)


# %% [markdown]
# ## Stage 0 — Step 3: Run environment diagnostics  --> Decision & next action 
#
# Based on the diagnostic result, follow the instruction printed by the next cell.
#
# This checks whether required packages are installed and whether a kernel restart is needed.
# The next cell will create a `status` object used by the decision step.
#
#

# %%
if status.ok and not status.needs_restart:
    print('[SPMpy] ✅ Environment ready.')
    print('[Next] Continue to Stage-1 below (Data Loading).')

elif status.ok and status.needs_restart:
    print('[SPMpy] ✅ Environment updated.')
    print('[Action] Restart the kernel, then re-run Stage-0 in this notebook.')

elif status.inconclusive:
    print('[SPMpy] ⚠ Environment status inconclusive.')
    print('[Action] Run the diagnostic notebook:')
    print('        notebooks/env_check_v_2025Dec_30_revised.ipynb')
    print('[Then] Return here, restart kernel if needed, and re-run Stage-0.')

else:
    print('[SPMpy] ❌ Environment not ready.')
    if status.missing:
        print('Missing packages:')
        for m in status.missing:
            print('  -', m)
    print('[Action] Fix the environment, restart kernel, then re-run Stage-0.')


# %% [markdown]
# <a id="Stage1"></a>
# # Stage 1 — File Loading (SXM, 3DS)
# [⬆ Back to Navigation](#Navigation)
#
#
# Stage 1 loads **Nanonis files** and standardizes them into **`xarray.Dataset`** objects.
# - [Stage 1.0](#Stage1.0): select folder 
# - [Stage 1.1](#Stage1.1): 2D image data (`.sxm`) → `xarray.Dataset`
# - [Stage 1.2](#Stage1.2): GridSpectroscopy (`.3ds`) → `xarray.Dataset`
#
# **Important:** Stage 1 performs *loading only* (no plane fit, no flattening, no filtering).
# Processing functions will be organized separately under a data-processing module.
#

# %% [markdown]
# <a id="Stage1.0"></a>
# ### Imports for Stage 1.0
# [⬆ Back to Navigation](#Navigation)
#
# In this Quickstart, the I/O logic is not defined inline.
# Instead, we import the legacy-compatible I/O functions from the package:
# - `select_folder()` — GUI folder picker
# - `files_in_folder()` — folder inventory → DataFrame (**no `os.chdir()`**)
# - `img2xr()` — `.sxm` → `xarray.Dataset`
# - `grid2xr()` — `.3ds` → `xarray.Dataset` (used in Stage 1.2)
#
# This keeps the Quickstart focused on workflow, while the implementation lives in `spmpy/io/`.
#
# #### USE `spmpy_io_library_v0_1_2`
#
#

# %%
# I/O function set (paired .py lives in: spmpy/io/spmpy_io_library_v0_1.py)
from spmpy.io import spmpy_io_library_v0_1_2 as io

select_folder = io.select_folder
files_in_folder = io.files_in_folder
img2xr = io.img2xr
grid2xr = io.grid2xr


# %% [markdown]
# #### Stage 1.0 Step 1 — Select a working folder
#
# Run the next cell to pick a folder that contains your `.sxm` / `.3ds` files.
#

# %%
selected_folder = select_folder()
if selected_folder:
    print(f"Selected folder: {selected_folder}")
else:
    print("No folder selected.")


# %% [markdown]
# #### Stage 1.0 Step 2 — Inventory the folder as a DataFrame
#
# This creates a DataFrame inventory so you can reproducibly select files by name.
#
# **Note:** Because we do not use `os.chdir()`, the DataFrame includes a full `file_path` column.
# Use `file_path` when loading files, and define an explicit `output_dir` when saving results later.
#

# %%
folder_path = selected_folder
print(f"Selected folder: {folder_path}")

files_df = files_in_folder(folder_path)
files_df

# %% [markdown]
# <a id="Stage1.1"></a>
# ## Stage 1.1 — 2D Image Data Loading (`.sxm`)
# [⬆ Back to Navigation](#Navigation)
#
# 1. List files in the folder as a DataFrame (for reproducible selection).
# 2. Choose an `.sxm` file name from the table.
# 3. Load the file into an **`xarray.Dataset`** using `img2xr`.
#
# ~~**Why this workflow**~~
# ~~This is intentionally designed to support future workflows where you load **multiple files** and build a dataset collection in a consistent way.~~
#

# %% [markdown]
# #### Stage 1.1 Step 0 — Select an `.sxm` file from the inventory
#
# Pick a file name from the DataFrame. You can keep a list for future multi-file loading.
#

# %%
# List all SXM files
file_list = files_df[files_df.type=='sxm'].file_name
file_list

# %%
files_df[files_df.file_name.str.contains('x1_20251017_20008.')]

# %%
# Choose one file (edit as needed)
sxm_name = file_list.iloc[0] if len(file_list) else None
sxm_name = file_list.iloc[88] if len(file_list) else None
sxm_name = file_list.iloc[48] if len(file_list) else None
sxm_name

# %% [markdown]
# #### Stage 1.1 Step 1 — Load the SXM file into an `xarray.Dataset`
#
# No plotting is performed here. The returned `xarray.Dataset` is sufficient for validation.
# * Only directly measured SXM files are supported at this stage; extracted or post-processed SXM files are intentionally excluded from loading.

# %%
from pathlib import Path

if sxm_name is None:
    raise RuntimeError('No .sxm files found in the selected folder.')

# Prefer explicit file_path if provided by files_in_folder()
if 'file_path' in files_df.columns:
    sxm_path = Path(files_df.loc[files_df.file_name == sxm_name, 'file_path'].iloc[0])
else:
    sxm_path = Path(folder_path) / sxm_name

print('[SPMpy] Loading:', sxm_path)

ds_sxm = img2xr(str(sxm_path), center_offset=False)
ds_sxm

# %% [markdown]
# #### Stage 1.1 Step 2— Add experiment metadata (attrs)
#
# SPMpy keeps experiment context in `Dataset.attrs`. Edit the values below to match your experiment.
# These fields are user-defined and will be used later in analysis/plotting pipelines.
#

# %%
# Edit these values for your dataset
'''
ds_sxm.attrs['tip'] = 'PtIr'
ds_sxm.attrs['sample'] = 'Cu(111)'
ds_sxm.attrs['ref_a0_nm'] = 0.257
ds_sxm.attrs['temperature'] = '4.35K'
'''

# Edit these values for your dataset
ds_sxm.attrs['tip'] = 'PtIr'

ds_sxm.attrs['sample'] = 'Fe5GeTe2'
ds_sxm.attrs['ref_a0_nm'] = 0.404 
ds_sxm.attrs['temperature'] = '40 mK'

#ds_sxm.attrs['sample'] = 'HOPG'
#ds_sxm.attrs['ref_a0_nm'] = 0.254 
#ds_sxm.attrs['temperature'] = '300K'

'''
# Example alternative (commented):
# ds_sxm.attrs['tip'] = 'Ni'
# ds_sxm.attrs['sample'] = 'FeTeSe'
# ds_sxm.attrs['ref_a0_nm'] = 0.384
# ds_sxm.attrs['temperature'] = '40mK'
'''
ds_sxm

# %%
ds_sxm

# %% [markdown]
# ## End of Stage 1.1
#
# At this point you have a **2D SXM image** loaded as an **`xarray.Dataset`**.
#
# Next:
#
# - [**Stage 2:**](#Stage2) visualization and data-processing steps (plane fit / flattening) from a dedicated module
#

# %% [markdown]
# <a id="Stage1.2"></a>
#
# ## Stage 1.2 — GridSpectroscopy (.3ds) loading
#
#
# [⬆ Back to Navigation](#Navigation)
#
# This section loads a Nanonis GridSpectroscopy file (`.3ds`) and converts it into an `xarray.Dataset`.
#
# ### What you will do
# 1. Select a `.3ds` file name from the folder inventory (`files_df`).
# 2. Load it with `grid2xr()` using an explicit `file_path`.
# 3. Add experiment metadata to `ds_grid.attrs`.
#
# **Note:** This stage performs loading only. Processing (plane fit / flattening / filtering) belongs to a
# dedicated data-processing module (Stage 2).
#

# %%
# Select one or more .3ds files from the inventory
file_list_3ds = files_df[files_df.type == '3ds'].file_name
file_list_3ds

# %%
# Choose a single file for loading (edit as needed)
if len(file_list_3ds) == 0:
    raise RuntimeError('No .3ds files found in the selected folder.')

grid_name = file_list_3ds.iloc[0]
print('Selected .3ds file:', grid_name)

# %%
from pathlib import Path

# Prefer explicit file_path if provided by files_in_folder()
if 'file_path' in files_df.columns:
    grid_path = Path(files_df.loc[files_df.file_name == grid_name, 'file_path'].iloc[0])
else:
    grid_path = Path(folder_path) / grid_name

print('[SPMpy] Loading:', grid_path)

ds_grid = grid2xr(str(grid_path))
ds_grid

# %% [markdown]
# ### Step — Add experiment metadata (attrs)
#
# Edit the values below to match your experiment.
# These fields are intentionally user-defined and will be used later in analysis/plotting pipelines.
#

# %%
# Edit these values for your grid dataset
'''
ds_grid.attrs['tip'] = 'PtIr'
ds_grid.attrs['sample'] = 'Cu(111)'
ds_grid.attrs['ref_a0_nm'] = 0.255
ds_grid.attrs['temperature'] = '4.35K'
'''
# Edit these values for your dataset
ds_grid.attrs['tip'] = 'PtIr'
ds_grid.attrs['sample'] = 'Fe5GeTe2'
ds_grid.attrs['ref_a0_nm'] = 0.404 
ds_grid.attrs['temperature'] = '4.35K'

# Example alternative (commented):
# ds_grid.attrs['tip'] = 'Ni'
# ds_grid.attrs['sample'] = 'FeTeSe'
# ds_grid.attrs['ref_a0_nm'] = 0.384
# ds_grid.attrs['temperature'] = '40mK'

ds_grid

# %%
ds_grid.isel(X=slice(0, 20), Y=slice(0, 20)).to_netcdf("ds_grid.nc")

# %% [markdown]
# ## End of Stage 1.2
#
# At this point you have:
#
#
# - `ds_grid`: a GridSpectroscopy dataset loaded as an `xarray.Dataset`
#
# Next (planned):
#
# - [**Stage 2**](#Stage2): Visualization and data-processing steps (plane fit / flattening) from a dedicated module.
#

# %% [markdown]
# <a id="Stage2"></a>
#
# # Stage 2 : Data-processing & visualization 
#
# [⬆ Back to Navigation](#Navigation)

# %%

# %% [markdown]
#
# <a id="Stage2.1"></a>
# ## Stage 2.1 — `ds_sxm` preprocessing
#
# [⬆ Back to Navigation](#Navigation)
#
# * interpolate2D
# * plane_fit_xr
# * plateau_tilt_xr

# %%
from spmpy.preprocess.interpolate2D_xr import interpolate2D_xr 

from spmpy.preprocess.plane_fit_xr import plane_fit_xr 
from spmpy.preprocess.plateau_tilt_xr import plateau_tilt_xr 




# %%

ds_sxm = interpolate2D_xr(ds_sxm)

# %%
ds_sxm

# %% [markdown]
# ## import plane_fit_xr & plateau_tilt_xr

# %%
ds_sxm_1 = plateau_tilt_xr(plane_fit_xr (ds_sxm,
                                         method='y_fit',
                                         poly_order=3,overwrite=True),
                           grad_sigma = 20,
                           overwrite=True)

# %%

# %%
plane_fit_xr(
    ds_sxm_1,
    ch='Z_P1_fwd',
    method='y_fit',
    remove_line_mean=True,
    remove_mean=True,
    poly_order=2,
    overwrite=True
).Z_P1_fwd.plot()


# %%
#ds_sxm.Z_fwd.plot()
#ds_sxm_1.Z_fwd.plot()
ds_sxm_1.Z_P1_fwd.plot()


# %%
import seaborn_image as isns
isns.imshow(ds_sxm_1.Z_P1_fwd)

# %% [markdown]
# ## import twoDfft_xr

# %%
from spmpy.fft.twoDfft_xrft import twoDfft_xrft,twoDifft_xrft

# %%
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



# %%
ds_sxm_fft = twoDfft_xrft(ds_sxm_1)

# %%
ds_fft = twoDfft_xrft(
    ds_sxm,
    ch=["P1"],
    detrend="linear",
    window="hann",
)

# %%
ds_fft.Z_P1_fwd_fft_amp.plot(norm=LogNorm())



# %%
ds_sxm_fft

# %%
from matplotlib.colors import LogNorm
#ds_sxm_fft.Z_fwd_fft_amp.plot(norm=LogNorm())
ds_sxm_fft.Z_P1_fwd_fft_amp.plot(norm=LogNorm())



# %%

# %%

# %%
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



# %%
ds_sxm_fft

# %%
ds_sxm_fft_ifft = twoDifft_xrft(ds_sxm_fft)

# %%
ds_sxm_fft_ifft

# %%
ds_sxm_fft_ifft.Z_P1_bwd_ifft.plot()

# %%
ds_sxm_1

# %%
import numpy as np
np.save('test_array',ds_sxm.Z_fwd.to_numpy())

# %%

# %%

# %%

# %% [markdown]
# ## if plateau tilt correction is suspicous, check the plateau mask 

# %%
ds_sxm_1= plane_fit_xr (ds_sxm,   
              ch ='Z_P1_fwd',
              method='y_fit',
              poly_order=3,
              overwrite=True)

# %%
ds_sxm_1= plateau_tilt_xr(ds_sxm_1, 
                ch ='Z_P1_fwd',grad_sigma = 20,
                          store_plateau_mask=True,
                overwrite=True)

# %%
#ds_sxm_1

# %%
import seaborn_image as isns
#isns.imshow(ds_sxm_1.Z_P1_fwd)
isns.imshow(ds_sxm_1.Z_P1_fwd_plateau_mask)


# %%
isns.imshow(ds_sxm_1.Z_P1_fwd)


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


