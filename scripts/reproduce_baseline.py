"""Thin entry point for the development-only baseline preflight."""

from reliable_endo_gs.baseline.reproduction import main, reproduce_baseline

__all__ = ["main", "reproduce_baseline"]


if __name__ == "__main__":
    raise SystemExit(main())
