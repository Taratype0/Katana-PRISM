"""Allow `python -m splicetrace` execution."""
from splicetrace.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
