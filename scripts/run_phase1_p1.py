"""Thin entry point for the separately authorized development-only P1 run."""

from reliable_endo_gs.evaluation.p1_runner import main


if __name__ == "__main__":
    raise SystemExit(main())
