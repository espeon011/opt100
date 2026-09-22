# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "vl-convert-python==1.9.0.post1",
# ]
# ///

# marimo が export した ipynb の altair 出力を PNG に置き換える.
# GitHub のノートブック表示は JavaScript を実行しないので, これらはそのままだと表示されない.

import base64
import json
import re
import sys

import vl_convert as vlc

WIDTH = 1000

VEGALITE_RE = re.compile(r"application/vnd\.vegalite\.v\d+\+json")
VEGA_RE = re.compile(r"application/vnd\.vega\.v\d+\+json")


def as_json(value):
    if isinstance(value, (dict, list)) and not (
        isinstance(value, list) and all(isinstance(v, str) for v in value)
    ):
        return value
    return json.loads("".join(value))


def fix_width(spec: dict) -> dict:
    if spec.get("width") == "container":
        spec = dict(spec, width=WIDTH)
    return spec


def to_png(data: dict) -> bytes | None:
    for mime, value in data.items():
        if VEGALITE_RE.fullmatch(mime) or VEGA_RE.fullmatch(mime):
            # mo.ui.altair_chart は中身が Vega-Lite でも vega の MIME で出てくるので $schema で判断する
            spec = fix_width(as_json(value))
            if "vega-lite" in spec.get("$schema", ""):
                return vlc.vegalite_to_png(spec, scale=1)
            return vlc.vega_to_png(spec, scale=1)
    return None


def convert(path: str) -> int:
    with open(path) as f:
        nb = json.load(f)

    n = 0
    for cell in nb["cells"]:
        for output in cell.get("outputs", []):
            if output.get("output_type") not in ("display_data", "execute_result"):
                continue
            png = to_png(output.get("data", {}))
            if png is None:
                continue
            output["data"] = {"image/png": base64.b64encode(png).decode()}
            output["metadata"] = {}
            n += 1

    if n > 0:
        with open(path, "w") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
            f.write("\n")
    return n


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(f"{path}: {convert(path)} outputs converted")
