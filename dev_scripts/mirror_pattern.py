"""Mirror horizontally a pre-defined pattern."""

import sys


def main() -> int:
    """Execute the main routine."""
    pattern = [
        "######     #    ####",
        "######    ##   #####",
        "#####    ##    #####",
        "######    ##   #####",
    ]

    print("[")
    for row in pattern:
        reversed_row = "".join(reversed(row))
        print(f"{repr(reversed_row)},")
    print("]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
