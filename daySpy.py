#! python3.12
"""Compatibility entrypoint for the archived DaySpy live runner."""

from legacy.daySpy import main


if __name__ == "__main__":
    raise SystemExit(main())
