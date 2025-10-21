"""Simple model version utilities - get latest model by timestamp."""
from __future__ import annotations
from pathlib import Path
from typing import Optional


def get_latest_model_dir(base_dir: Path = Path("model")) -> Optional[Path]:
    """
    Get the most recent model directory by folder name (timestamp-based).
    
    Args:
        base_dir: Base model directory
        
    Returns:
        Path to latest model directory or None if no versions found
    """
    if not base_dir.exists():
        return None
    
    # Get all subdirectories
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    if not subdirs:
        return None
    
    # Sort by name (assuming timestamp format: YYYYMMDD_HHMMSS)
    # This works because lexicographic sort = chronological sort for timestamps
    subdirs.sort(reverse=True)
    
    return subdirs[0]


def get_model_dir(version: Optional[str] = None, base_dir: Path = Path("model")) -> Path:
    """
    Get model directory by version name or latest if not specified.
    
    Args:
        version: Version name (folder name) or None for latest
        base_dir: Base model directory
        
    Returns:
        Path to model directory
        
    Raises:
        ValueError: If version not found or no models exist
    """
    if version:
        # Use specified version
        model_path = base_dir / version
        if not model_path.exists():
            raise ValueError(f"Model version '{version}' not found in {base_dir}")
        return model_path
    else:
        # Get latest version
        latest = get_latest_model_dir(base_dir)
        if latest is None:
            raise ValueError(f"No model versions found in {base_dir}")
        return latest

