"""Private path utilities for quality review agent."""

from pathlib import Path


def mkdir_private(path: Path, root: Path) -> None:
    """mkdir -p with 0o700 on every component created under root.

    mkdir(mode=) applies only to the leaf; intermediates get the default umask.
    This walks back from the leaf and applies 0o700 to every created directory.

    Args:
        path: Directory to create
        root: Stop walking at this ancestor (never chmod at or above root)
    """
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    cur = path
    while cur != root and root in cur.parents:
        cur.chmod(0o700)
        cur = cur.parent
