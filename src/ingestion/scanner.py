"""Folder scanning and file-level change detection."""

import hashlib
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def compute_file_hash(path: Path) -> str:
    """Compute MD5 hash of a file's raw contents."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def scan_folder(path: Path, *, recursive: bool = True) -> list[Path]:
    """Discover all supported document files in a folder.

    Returns sorted list of paths to supported files.
    """
    if recursive:
        all_files = path.rglob("*")
    else:
        all_files = path.glob("*")

    files = [
        f for f in all_files
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    files.sort()
    return files
