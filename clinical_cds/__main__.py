import sys

from clinical_cds.cli import main


if __name__ == "__main__":
    try:
        exit_code = main()
    except (ConnectionError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        exit_code = 2
    raise SystemExit(exit_code)
