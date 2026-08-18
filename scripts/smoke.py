"""Delegate the standalone smoke script to the package CLI."""

import sys

from reliable_endo_gs.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main(["smoke", *sys.argv[1:]]))
