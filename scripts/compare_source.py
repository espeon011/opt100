#! /usr/bin/env python3

# 2 つの ipynb でセルのコードが一致するか調べる. 出力は見ない.
# 一致しなければ差分を表示して 1 を返す.

import difflib
import json
import sys


def lines(path: str) -> list[str]:
    with open(path) as f:
        nb = json.load(f)
    out = []
    for i, cell in enumerate(nb["cells"]):
        out.append(f"--- cell {i} ---\n")
        out += "".join(cell["source"]).splitlines(keepends=True)
        out.append("\n")
    return out


def main() -> int:
    expected, actual = (lines(p) for p in sys.argv[1:3])
    if expected == actual:
        return 0

    diff = "".join(
        difflib.unified_diff(
            actual, expected, fromfile=sys.argv[2], tofile="marimo export", n=2
        )
    )
    print(diff if diff.endswith("\n") else diff + "\n", end="")
    return 1


if __name__ == "__main__":
    sys.exit(main())
