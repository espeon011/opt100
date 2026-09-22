#! /usr/bin/env sh

# ノートブックと書き出し済みの ipynb でセルのコードが一致するか調べる
# 引数なしなら全ノートブック, 引数があればそのノートブックだけ
# ノートブックは実行しないのでソルバーや API は使わない

set -eu

script_dir=$(cd "$(dirname "$0")" && pwd)
root_dir=$(dirname "$script_dir")

if [ $# -eq 0 ]; then
    set -- "$root_dir"/notebooks/*/*.py
fi

tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT

status=0
for f in "$@"; do
    f=$(realpath "$f")
    d=$(dirname "$f")
    n=$(basename "$f" .py)
    ipynb="$d/__marimo__/$n.ipynb"

    if [ ! -f "$ipynb" ]; then
        echo "NG $ipynb がない"
        status=1
        continue
    fi

    uv run --isolated --no-project --quiet --with marimo --with nbformat \
        marimo export ipynb --sort top-down -f "$f" -o "$tmp_dir/$n.ipynb" > /dev/null

    if python3 "$script_dir/compare_source.py" "$tmp_dir/$n.ipynb" "$ipynb"; then
        echo "OK $ipynb"
    else
        echo "NG $ipynb が $f と一致しない. scripts/export.sh で書き出し直す"
        status=1
    fi
done

exit $status
