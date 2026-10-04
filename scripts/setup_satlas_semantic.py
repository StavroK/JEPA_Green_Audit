"""Prepare the pinned Satlas upstream used by the optional semantic branch.

This script only clones/checks out source code. Model weights remain outside git.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

REPO = "https://github.com/allenai/satlas.git"
COMMIT = "c8b9aa5d4acdd3e4f58eb7cbb28ac18bb12c985f"
DEFAULT_DEST = Path("vendor/satlas")


def run(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()

    if not args.dest.exists():
        args.dest.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--filter=blob:none", REPO, str(args.dest))

    run("git", "fetch", "origin", COMMIT, cwd=args.dest)
    run("git", "checkout", "--detach", COMMIT, cwd=args.dest)

    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.dest, text=True
    ).strip()
    if actual != COMMIT:
        raise RuntimeError(f"Expected {COMMIT}, got {actual}")

    print(f"Satlas pinned at {actual}")
    print("Next: install optional semantic dependencies and download model weights.")


if __name__ == "__main__":
    main()
