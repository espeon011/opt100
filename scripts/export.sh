#! /usr/bin/env sh

# ノートブックを実行して ipynb を書き出し, altair の出力を PNG にする
# 引数なしなら全ノートブック, 引数があればそのノートブックだけ

set -eu

script_dir=$(cd "$(dirname "$0")" && pwd)
root_dir=$(dirname "$script_dir")

if [ $# -eq 0 ]; then
    set -- "$root_dir"/notebooks/*/*.py
fi

# jobshop.py は .env をカレントディレクトリから探すのでルートで実行する
for f in "$@"; do
    f=$(realpath "$f")
    d=$(dirname "$f")
    n=$(basename "$f" .py)
    (
        cd "$root_dir"
        marimo export ipynb --sandbox --include-outputs --sort top-down -f "$f" -o "$d/__marimo__/$n.ipynb"
        uv run "$script_dir/static_outputs.py" "$d/__marimo__/$n.ipynb"
    )
done
