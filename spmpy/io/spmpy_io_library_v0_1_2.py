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

# ## SPMpy I/O Library v0.1.2 (Notebook-paired)
#
# This notebook contains the **I/O function set** intended to live under `spmpy/io/`.
# It is designed to be **paired with a `.py` file via jupytext**.
#
# ## Goals
# - Preserve the legacy workflow interface where it makes sense.
# - Avoid hidden global side effects (especially `os.chdir()`).
# - Keep I/O responsibilities limited to: **read + standardize to `xarray.Dataset`**.
#
# ## Included functions
# - `select_folder()` — GUI folder picker (PyQt5)
# - `files_in_folder()` — inventory a folder into a `pandas.DataFrame` (**no `chdir`**)
# - `img2xr()` — load `.sxm` into `xarray.Dataset` (NetCDF-safe attrs)
#

# ## Why we avoid `os.chdir()`
#
# `os.chdir()` changes the **process-wide current working directory**. In a notebook workflow, this can silently
# affect unrelated cells and libraries that use relative paths.
#
# ### The design used here
# - We keep your **working folder** as an explicit variable, e.g. `folder_path`.
# - We store **full paths** for each file in the inventory DataFrame (`file_path`).
#
# ### Implication for saving results
# Yes, this means that **saving should also use explicit paths**. For example:
#
# - If you want outputs to go next to the raw data: use `output_dir = folder_path`.
# - If you want a clean separation: use `output_dir = Path(folder_path) / 'processed'`.
#
# In other words, you choose the target folder once (explicitly), then every save uses that folder.
# This is more reproducible than relying on whatever the current working directory happens to be.
#

# ## Imports
#
# These are standard dependencies for the I/O layer.
# (`nanonispy` is required only when `img2xr()` is called.)
#

# +
from __future__ import annotations

from pathlib import Path
import os
import glob
import json
import math
import re

import numpy as np
import pandas as pd
import xarray as xr

# Optional GUI dependency (only needed when select_folder() is used)
try:
    from PyQt5.QtWidgets import QApplication, QFileDialog
except Exception:
    QApplication = None
    QFileDialog = None

# -

# ## `select_folder()`
#
# Folder picker used in Quickstart Stage-1.
#

def select_folder() -> str:
    """Open a folder selection dialog and return the selected folder path.

    Returns
    -------
    str
        Selected folder path. Empty string if no folder was selected.
    """
    if QApplication is None or QFileDialog is None:
        raise ModuleNotFoundError(
            "PyQt5 is required for select_folder(). Install PyQt5 or use a non-GUI path workflow."
        )

    app = QApplication.instance()
    if app is None:
        import sys
        app = QApplication(sys.argv)

    file_dialog = QFileDialog()
    folder_path = file_dialog.getExistingDirectory(None, "Select Folder")
    return str(folder_path) if folder_path else ""



# ## `files_in_folder()` (no `chdir`)
#
# This function inventories a folder and returns a DataFrame with the **same columns** as your legacy workflow:
#
# - `group`, `num`, `file_name`, `type`
#
# Additionally, it includes two columns that make multi-folder workflows safer:
#
# - `folder_path` — the folder that was scanned
# - `file_path` — full path to each file
#
# Because `file_path` is explicit, later stages can load and save deterministically without relying on `os.chdir()`.
#

def files_in_folder(path_input: str, print_all: bool = False) -> pd.DataFrame:
    """Generate a DataFrame listing files in the specified folder.

    Parameters
    ----------
    path_input : str
        Folder path.
    print_all : bool, optional
        If True, prints the full DataFrame.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: ['group', 'num', 'file_name', 'type', 'folder_path', 'file_path'].
    """
    folder = Path(path_input)
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    # Keep this informational print (legacy-friendly) WITHOUT changing CWD.
    print("Current Path =", os.getcwd())
    print("Target Folder =", str(folder))

    # Inventory by extension (no chdir)
    sxm_files  = sorted([p.name for p in folder.glob('*.sxm')])
    grid_files = sorted([p.name for p in folder.glob('*.3ds')])
    csv_files  = sorted([p.name for p in folder.glob('*.csv')])
    gwy_files  = sorted([p.name for p in folder.glob('*.gwy')])
    xlsx_files = sorted([p.name for p in folder.glob('*.xlsx')])
    nc_files   = sorted([p.name for p in folder.glob('*.nc')])

    def _df_for(files, ext_len, num_slice=True):
        rows = []
        for fn in files:
            if num_slice:
                group = fn[:-7]
                num = fn[-7:-4]
            else:
                group = fn[:-ext_len]
                num = np.nan
            rows.append([group, num, fn])
        return pd.DataFrame(rows, columns=['group', 'num', 'file_name'])

    file_list_sxm_df  = _df_for(sxm_files,  4, num_slice=True)
    file_list_3ds_df  = _df_for(grid_files, 4, num_slice=True)
    file_list_csv_df  = _df_for(csv_files,  4, num_slice=True)
    file_list_gwy_df  = _df_for(gwy_files,  4, num_slice=False)
    file_list_xlsx_df = _df_for(xlsx_files, 5, num_slice=False)
    file_list_nc_df   = _df_for(nc_files,   3, num_slice=False)

    file_list_df = pd.concat(
        [file_list_sxm_df, file_list_3ds_df, file_list_csv_df,
         file_list_gwy_df, file_list_xlsx_df, file_list_nc_df],
        ignore_index=True
    )

    file_list_df['type'] = [fn[-3:] for fn in file_list_df.file_name]
    file_list_df.loc[file_list_df.type == 'lsx', 'type'] = 'xlsx'
    file_list_df.loc[file_list_df.type == '.nc', 'type'] = 'nc'

    # Add explicit paths
    file_list_df['folder_path'] = str(folder)
    file_list_df['file_path'] = [str(folder / fn) for fn in file_list_df.file_name]

    if print_all:
        print(file_list_df)

    # Legacy-style summary prints
    sxm_file_groups = list(set(file_list_sxm_df['group']))
    for group in sxm_file_groups:
        print('sxm file groups:', group, ': # of files =',
              len(file_list_sxm_df[file_list_sxm_df['group'] == group]))

    if len(file_list_df[file_list_df['type'] == '3ds']) == 0:
        print('No GridSpectroscopy data')
    else:
        print('# of GridSpectroscopy',
              list(set(file_list_df[file_list_df['type'] == '3ds'].group))[0],
              '=', file_list_df[file_list_df['type'] == '3ds'].group.count())

    return file_list_df



# ## `grid2xr()` (3DS → `xarray.Dataset`)
#
# This function loads Nanonis GridSpectroscopy `.3ds` files and standardizes them into an `xarray.Dataset`.
#
# Design rules:
# - I/O only: reading + metadata/coords standardization.
# - No plane fit / flattening / filtering here.
#
# If `nanonispy` (or other required dependencies) are missing, the function should raise a clear error.
#

def grid2xr(griddata_file: str, center_offset: bool = True) -> xr.Dataset:
    """
    Convert a Nanonis `.3ds` grid spectroscopy file into an `xarray.Dataset`.

    This function reads the grid file, constructs spatial coordinates (X, Y) and the
    bias axis (`bias_mV`), and exports spectroscopy signals into a labeled dataset.

    Key features (kept and clarified from the legacy implementation):
    - X/Y coordinates are created from the scan center and ROI size (in meters in the header).
    - Bias axis is stored in millivolts as `bias_mV`.
    - Forward/backward traces are kept if present (depending on the recorded channels).
    - Metadata is stored in `grid_xr.attrs`, including:
        * title (requested format)
        * image_size, X_spacing, Y_spacing
        * optional tip/sample/temperature placeholders (legacy convenience)

    Parameters
    ----------
    griddata_file
        Path to a Nanonis `.3ds` file.
    center_offset
        If True (default), X and Y coordinates remain centered around the scan center.
        If False, X and Y coordinates are shifted so that (X, Y) starts at the ROI lower-left corner.

    Returns
    -------
    xr.Dataset
        Dataset with dimensions:
        - Y, X: real-space coordinates (meters)
        - bias_mV: bias axis in millivolts

    Notes
    -----
    Title convention (as requested):
      - includes file name
      - includes ROI size in nm x nm
      - includes set bias (mV) and setpoint current (pA)
      - includes rotation if available in the header: "R=30deg"
    """
    import nanonispy as nap

    file = griddata_file
    NF = nap.read.NanonisFile(file)
    Gr = nap.read.Grid(NF.fname)

    # ----------------------------- parsing helpers -----------------------------
    _num_re = re.compile(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

    def parse_signed_float(v) -> float:
        """Parse a float from header values that may include units."""
        if v is None:
            return float("nan")
        if isinstance(v, (int, float, np.number)):
            return float(v)
        if isinstance(v, (list, tuple)) and len(v) > 0:
            v = v[0]
        s = str(v)
        m = _num_re.search(s)
        if not m:
            return float("nan")
        return float(m.group(1))

    def get_case_insensitive_key(d: dict, candidates: list[str]) -> str | None:
        """Return the first matching key in d for any candidate key (case-insensitive)."""
        if not isinstance(d, dict):
            return None
        low = {str(k).lower(): k for k in d.keys()}
        for c in candidates:
            if c.lower() in low:
                return low[c.lower()]
        return None

    # ----------------------------- geometry / coordinates -----------------------------
    dim_px, dim_py = Gr.header["dim_px"]
    cntr_x, cntr_y = Gr.header["pos_xy"]          # meters
    size_x, size_y = Gr.header["size_xy"]         # meters
    step_dx, step_dy = size_x / dim_px, size_y / dim_py

    x = np.linspace(cntr_x - size_x / 2, cntr_x + size_x / 2, dim_px)
    y = np.linspace(cntr_y - size_y / 2, cntr_y + size_y / 2, dim_py)

    # ----------------------------- spectroscopy axes -----------------------------
    bias = Gr.signals["sweep_signal"]  # Volts (typical)
    bias_mV = 1000.0 * np.array(bias, dtype=float)

    # ----------------------------- signals -----------------------------
    topography = Gr.signals["topo"]
    params_v = Gr.signals["params"]  # (dim_px, dim_py, 15) in legacy format

    # Build dataset with standard dims order (Y, X, bias)
    grid_xr = xr.Dataset(
        coords=dict(
            X=("X", x),
            Y=("Y", y),
            bias_mV=("bias_mV", bias_mV),
        )
    )

    # Nanonis stores arrays often as (X, Y, ...) so we transpose carefully to (Y, X, ...)
    # Topography is usually (dim_px, dim_py)
    grid_xr["topography"] = xr.DataArray(np.array(topography).T, dims=("Y", "X"), coords={"Y": y, "X": x})

    # Store params as a variable for completeness (legacy behavior)
    grid_xr["params"] = xr.DataArray(np.array(params_v).transpose(1, 0, 2), dims=("Y", "X", "params_dim"),
                                     coords={"Y": y, "X": x})

    # Add all spectroscopy channels (excluding topo/params/sweep_signal)
    for ch_name in Gr.signals.keys():
        if ch_name in ("topo", "params", "sweep_signal"):
            continue
        arr = np.array(Gr.signals[ch_name])
        # Expected shape: (dim_px, dim_py, n_bias) -> transpose to (dim_py, dim_px, n_bias)
        if arr.ndim == 3:
            arr2 = arr.transpose(1, 0, 2)
            grid_xr[ch_name] = xr.DataArray(arr2, dims=("Y", "X", "bias_mV"),
                                            coords={"Y": y, "X": x, "bias_mV": bias_mV})
        elif arr.ndim == 2:
            arr2 = arr.T
            grid_xr[ch_name] = xr.DataArray(arr2, dims=("Y", "X"), coords={"Y": y, "X": x})
        else:
            # Keep any unexpected shape as-is with minimal labeling
            grid_xr[ch_name] = xr.DataArray(arr)

    # ----------------------------- coordinate offset convention -----------------------------
    if not center_offset:
        grid_xr = grid_xr.assign_coords(
            X=(grid_xr["X"] + (cntr_x - size_x / 2.0)),
            Y=(grid_xr["Y"] + (cntr_y - size_y / 2.0)),
        )

    # ----------------------------- title (requested format) -----------------------------
    basename = os.path.basename(file)
    roi_x_nm = float(size_x) * 1e9
    roi_y_nm = float(size_y) * 1e9

    k_bias = get_case_insensitive_key(Gr.header, ["bias>bias (v)", "bias>bias (V)", "bias"])
    k_setpt = get_case_insensitive_key(Gr.header, ["z-controller>setpoint", "z-controller>setpoint (A)", "setpoint"])
    V_set = parse_signed_float(Gr.header.get(k_bias)) if k_bias else float("nan")
    I_set = parse_signed_float(Gr.header.get(k_setpt)) if k_setpt else float("nan")
    if np.isnan(V_set):
        V_set = 0.0
    if np.isnan(I_set):
        I_set = 0.0

    bias_set_mV = 1000.0 * float(V_set)
    setpoint_pA = 1e12 * float(I_set)

    # Optional rotation (header-dependent): search for a plausible rotation/angle key
    rot_deg = None
    for key in Gr.header.keys():
        lk = str(key).lower()
        if "scan_angle" in lk or ("scan" in lk and "angle" in lk) or "rotation" in lk:
            try:
                rot_deg = float(parse_signed_float(Gr.header.get(key)))
                break
            except Exception:
                pass

    title = f"{basename}\n" \
            f"{round(roi_x_nm)} nm x {round(roi_y_nm)} nm  " \
            f"V = {bias_set_mV:.2f} mV, I = {round(setpoint_pA)} pA"
    if rot_deg is not None and not math.isclose(rot_deg, 0.0):
        title += f"  R = {int(round(rot_deg))}deg"

    # Keep sweep range as a second line (useful for grids)
    try:
        sweep_start = float(Gr.header.get("Bias Spectroscopy>Sweep Start (V)"))
        sweep_end = float(Gr.header.get("Bias Spectroscopy>Sweep End (V)"))
        title += f"\nSweep: {sweep_start:.3f} V → {sweep_end:.3f} V"
    except Exception:
        pass

    # ----------------------------- attrs -----------------------------
    grid_xr.attrs["title"] = title
    grid_xr.attrs["image_size"] = [float(size_x), float(size_y)]
    grid_xr.attrs["X_spacing"] = float(step_dx)
    grid_xr.attrs["Y_spacing"] = float(step_dy)
    grid_xr.attrs["data_vars_list"] = list(grid_xr.data_vars.keys())

    # Legacy convenience placeholders (safe defaults)
    if "tip" not in grid_xr.attrs:
        grid_xr.attrs["tip"] = "To Be Announced"
    if "sample" not in grid_xr.attrs:
        grid_xr.attrs["sample"] = "To Be Announced"
    if "temperature" not in grid_xr.attrs:
        grid_xr.attrs["temperature"] = "To Be Announced"

    return grid_xr


# ## `img2xr()` (SXM → `xarray.Dataset`)
#
# This is the same `img2xr_updated` logic you provided (robust multipass detection + NetCDF-safe attrs),
# kept as an I/O-only function.
#
# Stage-0 should ensure dependencies are available. If `nanonispy` is missing, this function raises a clear error.
#

def img2xr(loading_sxm_file: str, center_offset: bool = True) -> xr.Dataset:
    """
    Convert a Nanonis `.sxm` image file into an `xarray.Dataset`.

    This function is intended to be an I/O utility: it reads a Nanonis image,
    builds physically meaningful coordinates (X, Y in meters), and stores
    scan metadata in `ds.attrs` in a NetCDF-safe manner.

    Key features (restored from the 2025-12-04 implementation):
    - Robust multipass detection (header-based and signal-name-based).
    - Multipass channel reconstruction into variables like:
        * Z_P1_fwd, Z_P1_bwd, LIX_P1_fwd, LIX_P1_bwd, ...
      along with `channels_index` and `bias_table` in attributes.
    - Strict LIX detection:
        * Only signals containing both "LI" and "X" (case-insensitive) are treated as LIX.
        * CURRENT is *not* used as a fallback for LIX.
        * CURRENT (if present) is stored separately as CURR_fwd / CURR_bwd.
    - NetCDF-safe attribute serialization: dict/list attributes are JSON-serialized.

    Parameters
    ----------
    loading_sxm_file
        Path to a Nanonis `.sxm` file.
    center_offset
        If False (default), X and Y coordinates are shifted so that the origin is at the lower-left
        corner of the ROI (i.e., Scan offset is applied: `cntr - size/2`).
        If True, X and Y coordinates remain centered around the scan center (i.e., no additional shift).

    Returns
    -------
    xr.Dataset
        Dataset with:
        - coords: X (m), Y (m)
        - data_vars: image channels (Z, LIX, LIY, CURR, etc.) in forward/backward directions
        - attrs: metadata including title, ROI size, spacing, rotation, multipass tables, etc.

    Notes
    -----
    Title convention (as requested):
      - includes file name
      - includes ROI size in nm x nm
      - includes set bias (mV) and setpoint current (pA)
      - includes rotation if present: "R=30deg"
    """
    try:
        import nanonispy as nap
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "nanonispy is required for reading .sxm files. "
            "Please install dependencies (Stage-0) and restart the kernel."
        ) from e

    # ----------------------------- NetCDF-safe attribute sanitizer -----------------------------
    def _sanitize_attr_value(v):
        """Convert values into NetCDF-friendly primitives (or JSON strings)."""
        if isinstance(v, (bool, np.bool_)):
            return int(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (list, tuple)):
            try:
                return json.dumps(v)
            except TypeError:
                return json.dumps([str(x) for x in v])
        if isinstance(v, dict):
            try:
                return json.dumps(v)
            except TypeError:
                return json.dumps({str(k): str(val) for k, val in v.items()})
        return v

    def _sanitize_dataset_attrs(ds: xr.Dataset) -> xr.Dataset:
        """Return a copy of the dataset with NetCDF-safe attrs."""
        ds = ds.copy()
        clean = {}
        for k, v in ds.attrs.items():
            clean[str(k)] = _sanitize_attr_value(v)
        ds.attrs = clean
        return ds

    # ----------------------------- parsing helpers -----------------------------
    _num_re = re.compile(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

    def parse_signed_float(v, assume_unit: str = "") -> float:
        """
        Parse a numeric value from a header entry.

        Header values may appear as:
          - float/int
          - strings like "100 mV", "1.0e-10 A", "200pA"
          - or lists/tuples where the first entry is the value string.

        Units are *not* fully normalized here; the caller chooses a unit interpretation.
        """
        if v is None:
            return float("nan")
        if isinstance(v, (int, float, np.number)):
            return float(v)
        if isinstance(v, (list, tuple)) and len(v) > 0:
            v = v[0]
        s = str(v)
        m = _num_re.search(s)
        if not m:
            return float("nan")
        return float(m.group(1))

    def get_case_insensitive_key(d: dict, candidates: list[str]) -> str | None:
        """Return the first matching key in d for any candidate key (case-insensitive)."""
        if not isinstance(d, dict):
            return None
        low = {str(k).lower(): k for k in d.keys()}
        for c in candidates:
            if c.lower() in low:
                return low[c.lower()]
        return None

    def as_float_scalar(x) -> float:
        """Return a python float from an array scalar / numpy scalar / python numeric."""
        try:
            return float(np.asarray(x).reshape(-1)[0])
        except Exception:
            return float(x)

    def classify_signal_name(sig_name: str) -> tuple[str, int | None]:
        """
        Classify a Nanonis signal name into a (kind, pass_index).

        kind is one of: "Z", "LI_X", "LI_Y", "Current", "Other".
        pass_index is parsed from "...P1..." patterns (None if not present).
        """
        sk = str(sig_name).upper()
        m = re.search(r"P\s*(\d+)", sk)
        pidx = int(m.group(1)) if m else None
        if "Z" in sk and "LI" not in sk:
            kind = "Z"
        elif "LI" in sk and "X" in sk:
            kind = "LI_X"
        elif "LI" in sk and "Y" in sk:
            kind = "LI_Y"
        elif "CURRENT" in sk:
            kind = "Current"
        else:
            kind = "Other"
        return kind, pidx

    # ----------------------------- open file + tolerant header access -----------------------------
    NF = nap.read.NanonisFile(loading_sxm_file)
    Scan = nap.read.Scan(NF.fname)

    # Bias / setpoint parsing (robust to small header variations)
    k_bias = get_case_insensitive_key(Scan.header, ["bias>bias (v)", "bias>bias (V)", "bias"])
    k_setpt = get_case_insensitive_key(Scan.header, ["z-controller>setpoint", "z-controller>setpoint (A)", "setpoint"])

    V_b = parse_signed_float(Scan.header[k_bias], assume_unit="V") if k_bias else np.nan
    I_t = parse_signed_float(Scan.header[k_setpt], assume_unit="A") if k_setpt else np.nan
    if np.isnan(V_b):
        V_b = 0.0
    if np.isnan(I_t):
        I_t = 0.0

    size_x, size_y = Scan.header["scan_range"]        # meters
    cntr_x, cntr_y = Scan.header["scan_offset"]       # meters
    dim_px, dim_py = Scan.header["scan_pixels"]       # pixels
    step_dx, step_dy = size_x / dim_px, size_y / dim_py
    Rot_Rad = math.radians(float(Scan.header["scan_angle"]))
    scan_dir = Scan.header.get("scan_dir", "up")
    basename = os.path.basename(getattr(Scan, "basename", NF.fname))

    # ----------------------------- multipass detection -----------------------------
    if "multipass-config" in Scan.header.keys():
        mp_cfg = Scan.header.get("multipass-config", {})
        is_multipass = True
    else:
        header_keys_lower = {str(k).lower(): k for k in Scan.header.keys()}
        mp_header_key = None
        for lk, orig in header_keys_lower.items():
            if "multipass" in lk:
                mp_header_key = orig
                break
        has_mp_cfg = mp_header_key is not None
        mp_cfg = Scan.header.get(mp_header_key, {}) if has_mp_cfg else {}
        if not isinstance(mp_cfg, dict):
            mp_cfg = {}

        # Fallback: if any signal name contains P<number>, treat as multipass
        has_p = any(re.search(r"P\s*\d+", str(k), flags=re.I) for k in Scan.signals.keys())
        is_multipass = bool(has_mp_cfg or has_p)

    # ----------------------------- multipass path -----------------------------
    if is_multipass:
        # Build bias overrides table if available
        bias_map: dict[tuple[int, str], float] = {}
        values = mp_cfg.get("Bias_override_value", [])
        if isinstance(values, (str, int, float)):
            values = [values]
        try:
            vals = [float(v) for v in values]
        except Exception:
            # Some files store bias override values as comma-separated strings
            vals = []
            for v in values:
                if v is None:
                    continue
                for tok in str(v).replace(";", ",").split(","):
                    tok = tok.strip()
                    if tok:
                        try:
                            vals.append(float(tok))
                        except Exception:
                            pass

        n_passes = len(vals) // 2 if len(vals) >= 2 else 0
        if n_passes > 0:
            for k in range(n_passes):
                bias_map[(k + 1, "forward")] = float(vals[2 * k + 0])
                bias_map[(k + 1, "backward")] = float(vals[2 * k + 1])

        # Coordinates (centered by construction; shift later if requested)
        X = np.linspace(-size_x / 2.0, size_x / 2.0, dim_px)
        Y = np.linspace(-size_y / 2.0, size_y / 2.0, dim_py)
        ds = xr.Dataset(coords={"X": X, "Y": Y})

        # Reconstruct channels:
        # - Forward scan: store as-is
        # - Backward scan: reverse fast-axis to match forward orientation
        for sig_name in Scan.signals.keys():
            kind, pidx = classify_signal_name(sig_name)
            if pidx is None:
                # Multipass signals should include pass index; keep unknowns if present
                pidx = 1
            sig = Scan.signals[sig_name]
            for direction in ("forward", "backward"):
                if direction not in sig:
                    continue
                arr = np.array(sig[direction])
                if direction == "backward":
                    arr = arr[:, ::-1]
                var = None
                if kind == "Z":
                    var = f"Z_P{pidx}_{'fwd' if direction=='forward' else 'bwd'}"
                elif kind == "LI_X":
                    var = f"LIX_P{pidx}_{'fwd' if direction=='forward' else 'bwd'}"
                elif kind == "LI_Y":
                    var = f"LIY_P{pidx}_{'fwd' if direction=='forward' else 'bwd'}"
                elif kind == "Current":
                    var = f"CURR_P{pidx}_{'fwd' if direction=='forward' else 'bwd'}"
                else:
                    # Keep other channels but make names NetCDF-safe-ish
                    safe = re.sub(r"[^0-9a-zA-Z_]+", "_", str(sig_name)).strip("_")
                    var = f"{safe}_P{pidx}_{'fwd' if direction=='forward' else 'bwd'}"
                ds[var] = xr.DataArray(arr, dims=("Y", "X"), coords={"Y": ds["Y"], "X": ds["X"]})

        # Enforce square pixel spacing by interpolation when needed (matches 2025-12-04 logic)
        if not np.isclose(step_dx, step_dy):
            ny, nx = ds.sizes["Y"], ds.sizes["X"]
            x0, x1 = as_float_scalar(ds["X"].values[0]), as_float_scalar(ds["X"].values[-1])
            y0, y1 = as_float_scalar(ds["Y"].values[0]), as_float_scalar(ds["Y"].values[-1])
            ratio = step_dy / step_dx
            ny_new = max(int(round(ny * ratio)), 1)
            X_new = np.linspace(float(x0), float(x1), int(nx))
            Y_new = np.linspace(float(y0), float(y1), int(ny_new))
            ds = ds.interp(X=X_new, Y=Y_new, method="linear")
            eff_dx = (float(X_new[-1]) - float(X_new[0])) / max(len(X_new) - 1, 1)
            eff_dy = (float(Y_new[-1]) - float(Y_new[0])) / max(len(Y_new) - 1, 1)
        else:
            eff_dx, eff_dy = step_dx, step_dy

        if not center_offset:
            ds = ds.assign_coords(
                X=(ds["X"] + (cntr_x - size_x / 2.0)),
                Y=(ds["Y"] + (cntr_y - size_y / 2.0)),
            )

        # ----------------------------- title (requested format) -----------------------------
        roi_x_nm = float(size_x) * 1e9
        roi_y_nm = float(size_y) * 1e9
        bias_mV = 1000.0 * float(V_b)
        setpoint_pA = 1e12 * float(I_t)
        title = (    f"{basename}\n"  
                     f"{round(roi_x_nm)} nm x {round(roi_y_nm)} nm, "
                     f"V = {bias_mV:.0f} mV, "
                     f"I = {round(setpoint_pA)} pA"
                )
        if not math.isclose(Rot_Rad, 0.0):
            title += f"  R={int(round(math.degrees(Rot_Rad)))}deg"

        # Multipass per-pass bias summary (if overrides exist)
        passes = sorted({int(m.group(1)) for v in ds.data_vars for m in [re.search(r"P(\d+)", v)] if m})
        bias_info = []
        for p in passes:
            bf_mV = 1000.0 * float(bias_map.get((p, "forward"), V_b))
            bb_mV = 1000.0 * float(bias_map.get((p, "backward"), V_b))
            bias_info.append(f"P{p} fwd @{bf_mV:.2f} mV / bwd @{bb_mV:.2f} mV")
        if bias_info:
            title += "\n" + " // ".join(bias_info)

        # Minimal tip/sample inference maintained for backward-compat with older notebooks
        def infer_tip(title_str: str) -> str:
            if "PTIR" in title_str.upper(): return "PtIr"
            if "WTIP" in title_str.upper(): return "W"
            if "CO_COATED" in title_str.upper(): return "Co_coated"
            if "AFM" in title_str.upper(): return "AFM"
            return "To Be Announced"

        def infer_sample(title_str: str) -> str:
            if "NBSE2" in title_str.upper(): return "NbSe2"
            if "CU(111)" in title_str.upper(): return "Cu(111)"
            if "AU(111)" in title_str.upper(): return "Au(111)"
            if "HOPG" in title_str.upper(): return "HOPG"
            return "To Be Announced"

        # ----------------------------- attrs (restored) -----------------------------
        ds.attrs["multipass"] = True
        ds.attrs["n_passes"] = int(len(passes)) if passes else 1
        ds.attrs["title"] = title
        ds.attrs["tip"] = infer_tip(title)
        ds.attrs["sample"] = infer_sample(title)
        ds.attrs["image_size"] = [float(size_x), float(size_y)]
        ds.attrs["X_spacing"] = float(eff_dx)
        ds.attrs["Y_spacing"] = float(eff_dy)

        ds.attrs["channels_index"] = [
            dict(
                var=vn,
                pass_index=(int(m.group(1)) if (m := re.search(r"P(\d+)", vn)) else None),
                dir=("forward" if vn.endswith("_fwd") else "backward" if vn.endswith("_bwd") else ""),
                kind=("Z" if vn.startswith("Z") else
                      "LI_X" if vn.startswith("LIX") else
                      "LI_Y" if vn.startswith("LIY") else
                      "Current" if vn.startswith("CURR") else
                      "Other"),
                bias_mV=(1000.0 * float(
                    (lambda p, d: (bias_map.get((p, "forward"), V_b) if d == "forward"
                                   else bias_map.get((p, "backward"), V_b)))
                    (int(m.group(1)) if (m := re.search(r"P(\d+)", vn)) else 1,
                     "forward" if vn.endswith("_fwd") else "backward")
                )),
            )
            for vn in ds.data_vars
        ]

        ds.attrs["bias_table"] = [
            dict(
                pass_index=int(p),
                bias_fwd_V=float(bias_map.get((p, "forward"), V_b)),
                bias_bwd_V=float(bias_map.get((p, "backward"), V_b)),
            )
            for p in passes
        ] if passes else [dict(pass_index=1, bias_fwd_V=float(V_b), bias_bwd_V=float(V_b))]

        ds.attrs["scan_angle_deg"] = float(math.degrees(Rot_Rad))
        ds.attrs["scan_dir"] = str(scan_dir)
        ds.attrs["data_vars_list"] = list(ds.data_vars.keys())

        ds = _sanitize_dataset_attrs(ds)
        return ds

    # ----------------------------- single-pass path -----------------------------
    z_fwd = np.array(Scan.signals["Z"]["forward"]) if "Z" in Scan.signals else None
    z_bwd = np.array(Scan.signals["Z"]["backward"])[:, ::-1] if "Z" in Scan.signals else None

    # STRICT LIX (no CURRENT fallback)
    lix_key = None
    for k in Scan.signals.keys():
        s = str(k).upper()
        if ("LI" in s) and ("X" in s):
            lix_key = k
            break
    lix_fwd = np.array(Scan.signals[lix_key]["forward"]) if lix_key and "forward" in Scan.signals[lix_key] else None
    lix_bwd = np.array(Scan.signals[lix_key]["backward"])[:, ::-1] if lix_key and "backward" in Scan.signals[lix_key] else None

    liy_key = None
    for k in Scan.signals.keys():
        s = str(k).upper()
        if ("LI" in s) and ("Y" in s):
            liy_key = k
            break
    liy_fwd = np.array(Scan.signals[liy_key]["forward"]) if liy_key and "forward" in Scan.signals[liy_key] else None
    liy_bwd = np.array(Scan.signals[liy_key]["backward"])[:, ::-1] if liy_key and "backward" in Scan.signals[liy_key] else None

    curr_key = None
    for k in Scan.signals.keys():
        if "CURRENT" in str(k).upper():
            curr_key = k
            break
    curr_fwd = np.array(Scan.signals[curr_key]["forward"]) if curr_key and "forward" in Scan.signals[curr_key] else None
    curr_bwd = np.array(Scan.signals[curr_key]["backward"])[:, ::-1] if curr_key and "backward" in Scan.signals[curr_key] else None

    # Base dataset (centered by construction; shift later if requested)
    X = np.linspace(-size_x / 2.0, size_x / 2.0, dim_px)
    Y = np.linspace(-size_y / 2.0, size_y / 2.0, dim_py)
    ds = xr.Dataset(coords={"X": X, "Y": Y})

    if z_fwd is not None:
        ds["Z_fwd"] = xr.DataArray(z_fwd, dims=("Y", "X"), coords={"Y": ds["Y"], "X": ds["X"]})
    if z_bwd is not None:
        ds["Z_bwd"] = xr.DataArray(z_bwd, dims=("Y", "X"), coords={"Y": ds["Y"], "X": ds["X"]})
    if lix_fwd is not None:
        ds["LIX_fwd"] = xr.DataArray(lix_fwd, dims=("Y", "X"), coords={"Y": ds["Y"], "X": ds["X"]})
    if lix_bwd is not None:
        ds["LIX_bwd"] = xr.DataArray(lix_bwd, dims=("Y", "X"), coords={"Y": ds["Y"], "X": ds["X"]})
    if liy_fwd is not None:
        ds["LIY_fwd"] = xr.DataArray(liy_fwd, dims=("Y", "X"), coords={"Y": ds["Y"], "X": ds["X"]})
    if liy_bwd is not None:
        ds["LIY_bwd"] = xr.DataArray(liy_bwd, dims=("Y", "X"), coords={"Y": ds["Y"], "X": ds["X"]})
    if curr_fwd is not None:
        ds["CURR_fwd"] = xr.DataArray(curr_fwd, dims=("Y", "X"), coords={"Y": ds["Y"], "X": ds["X"]})
    if curr_bwd is not None:
        ds["CURR_bwd"] = xr.DataArray(curr_bwd, dims=("Y", "X"), coords={"Y": ds["Y"], "X": ds["X"]})

    # Interpolate if pixel aspect ratio differs
    if not np.isclose(step_dx, step_dy):
        ny, nx = ds.sizes["Y"], ds.sizes["X"]
        x0, x1 = as_float_scalar(ds["X"].values[0]), as_float_scalar(ds["X"].values[-1])
        y0, y1 = as_float_scalar(ds["Y"].values[0]), as_float_scalar(ds["Y"].values[-1])
        ratio = step_dy / step_dx
        ny_new = max(int(round(ny * ratio)), 1)
        X_new = np.linspace(float(x0), float(x1), int(nx))
        Y_new = np.linspace(float(y0), float(y1), int(ny_new))
        ds = ds.interp(X=X_new, Y=Y_new, method="linear")
        eff_dx = (float(X_new[-1]) - float(X_new[0])) / max(len(X_new) - 1, 1)
        eff_dy = (float(Y_new[-1]) - float(Y_new[0])) / max(len(Y_new) - 1, 1)
    else:
        eff_dx, eff_dy = step_dx, step_dy

    # Apply center offset ONLY when requested
    # center_offset=True  -> shift to lower-left origin
    # center_offset=False -> keep centered coordinates
    if center_offset:
        ds = ds.assign_coords(
            X=(ds["X"] + (cntr_x - size_x / 2.0)),
            Y=(ds["Y"] + (cntr_y - size_y / 2.0)),
        )

    # Title (requested format)
    roi_x_nm = float(size_x) * 1e9
    roi_y_nm = float(size_y) * 1e9
    bias_mV = 1000.0 * float(V_b)
    setpoint_pA = 1e12 * float(I_t)
    base_title = (
        f"{basename}\n"
        f"{round(roi_x_nm)} nm x {round(roi_y_nm)} nm, "
        f"V = {bias_mV:.0f} mV, "
        f"I = {round(setpoint_pA)} pA"
    )
    if not math.isclose(Rot_Rad, 0.0):
        base_title += f"  R={int(round(math.degrees(Rot_Rad)))}deg"

    # Minimal tip/sample inference maintained for backward-compat with older notebooks
    def infer_tip(title_str: str) -> str:
        if "PTIR" in title_str.upper(): return "PtIr"
        if "WTIP" in title_str.upper(): return "W"
        if "CO_COATED" in title_str.upper(): return "Co_coated"
        if "AFM" in title_str.upper(): return "AFM"
        return "To Be Announced"

    def infer_sample(title_str: str) -> str:
        if "NBSE2" in title_str.upper(): return "NbSe2"
        if "CU(111)" in title_str.upper(): return "Cu(111)"
        if "AU(111)" in title_str.upper(): return "Au(111)"
        if "HOPG" in title_str.upper(): return "HOPG"
        return "To Be Announced"

    channels_index = []
    for vn in ds.data_vars:
        direction = "forward" if vn.endswith("_fwd") else "backward" if vn.endswith("_bwd") else ""
        if vn.startswith("Z"): kind = "Z"
        elif vn.startswith("LIX"): kind = "LI_X"
        elif vn.startswith("LIY"): kind = "LI_Y"
        elif vn.startswith("CURR"): kind = "Current"
        else: kind = "Other"
        channels_index.append(dict(var=vn, pass_index=1, dir=direction, kind=kind, bias_mV=1000.0 * float(V_b)))

    ds.attrs["multipass"] = False
    ds.attrs["n_passes"] = 1
    ds.attrs["title"] = base_title
    ds.attrs["tip"] = infer_tip(base_title)
    ds.attrs["sample"] = infer_sample(base_title)
    ds.attrs["image_size"] = [float(size_x), float(size_y)]
    ds.attrs["X_spacing"] = float(eff_dx)
    ds.attrs["Y_spacing"] = float(eff_dy)
    ds.attrs["channels_index"] = channels_index
    ds.attrs["bias_table"] = [dict(pass_index=1, bias_fwd_V=float(V_b), bias_bwd_V=float(V_b))]
    ds.attrs["scan_angle_deg"] = float(math.degrees(Rot_Rad))
    ds.attrs["scan_dir"] = str(scan_dir)
    ds.attrs["data_vars_list"] = list(ds.data_vars.keys())

    ds = _sanitize_dataset_attrs(ds)
    return ds


