"""Allow `python -m scsplicetrace` execution."""
from scsplicetrace.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
