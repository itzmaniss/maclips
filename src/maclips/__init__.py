"""maclips — semi-automated clipping for a single Apple Silicon Mac."""

__all__ = ["main"]


def main() -> int:
    """Console-script entry point; the real CLI lives in maclips.cli."""
    from .cli import main as _main

    return _main()
