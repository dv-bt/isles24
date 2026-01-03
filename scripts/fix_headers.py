"""
Fix image headers by snapping affine transforms if they're sufficiently close.
"""

from pathlib import Path
from isles.utils import snap_affines


def main():
    data_root = Path("/home/renku/work/data-local")
    log_file = data_root / "logs/snap-affine.log"
    atol = 1e-4

    snap_affines(
        data_root=data_root,
        modalities=None,
        atol=atol,
        log_file=log_file,
    )


if __name__ == "__main__":
    main()
