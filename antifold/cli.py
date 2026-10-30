"""Command-line entry point for AntiFold, equivalent to `python -m antifold.main`.

All options are handled by `antifold.main`, whose argument parsing and
logging live under its `__main__` guard, so run it as `__main__`.
"""

import runpy
import sys


def main():
    sys.argv[0] = "antifold"
    runpy.run_module("antifold.main", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
