"""UI helpers for notebooks (PyQt5-based).

Stage-1 purpose:
- Provide a simple folder selection dialog for STM users running Jupyter notebooks.

Notes
-----
- Requires PyQt5 (install via conda-forge is recommended).
- Creating multiple QApplication instances can cause issues; this function reuses an existing instance if present.
"""

from __future__ import annotations

import sys
from typing import Optional


def select_folder(title: str = "Select Folder") -> Optional[str]:
    """Open a native dialog to select a folder.

    Parameters
    ----------
    title : str
        Dialog title.

    Returns
    -------
    folder_path : str | None
        Selected folder path, or None if cancelled.
    """
    try:
        from PyQt5.QtWidgets import QApplication, QFileDialog
    except Exception as e:
        raise ImportError(
            "PyQt5 is required for folder selection. Install via conda-forge: conda install -c conda-forge pyqt"
        ) from e

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    file_dialog = QFileDialog()
    folder_path = file_dialog.getExistingDirectory(None, title)
    return folder_path if folder_path else None
