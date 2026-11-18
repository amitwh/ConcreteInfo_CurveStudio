"""Render the CurveStudio desktop icon: PNG sizes plus a multi-size ``.ico``.

Pillow cannot rasterize SVG, so this script re-draws the geometry of
``icon.svg`` (the canonical source) with Pillow primitives, supersampled 4x
and downscaled with Lanczos for smooth edges. The output is deterministic —
same Pillow, same bytes — which is why the binaries are generated at build
time (here and in CI) instead of being committed.

Usage (from anywhere; paths are absolute-safe):

    uv run python apps/desktop/packaging/make_icon.py [output-dir]

Writes ``curvestudio-desktop.png`` (256 px, for the Linux tarball and .desktop
install), ``icon-<size>.png`` for each size, and ``curvestudio-desktop.ico``
(multi-size, for the Windows executable and installer).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

CANONICAL = 512.0  # the viewBox of icon.svg; every coordinate below is scaled by size/512
BACKGROUND = (32, 36, 44, 255)  # #20242c
AXES = (232, 234, 240, 255)  # #e8eaf0
CURVE = (232, 115, 74, 255)  # #e8734a
POINTS = (154, 163, 178, 255)  # #9aa3b2

PNG_SIZES = (16, 24, 32, 48, 64, 128, 256, 512)
ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)
SUPERSAMPLE = 4

# icon.svg geometry, in viewBox units
_AXIS = ((112.0, 96.0), (112.0, 400.0), (416.0, 400.0))
_CURVE = ((136.0, 368.0), (224.0, 368.0), (264.0, 200.0), (400.0, 152.0))  # cubic bezier
_DOTS = ((168.0, 344.0), (248.0, 296.0), (336.0, 216.0))
_RADIUS = 96.0


def _bezier(control: tuple[tuple[float, float], ...], steps: int = 96) -> list[tuple[float, float]]:
    """Sample a cubic bezier curve at ``steps + 1`` uniform parameters."""
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = control
    points: list[tuple[float, float]] = []
    for i in range(steps + 1):
        t = i / steps
        u = 1.0 - t
        w0 = u * u * u
        w1 = 3.0 * u * u * t
        w2 = 3.0 * u * t * t
        w3 = t * t * t
        points.append(
            (w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3, w0 * y0 + w1 * y1 + w2 * y2 + w3 * y3)
        )
    return points


def render(size: int) -> Image.Image:
    """Draw the icon at ``size`` px (drawn 4x larger, then downscaled)."""
    scale = size * SUPERSAMPLE / CANONICAL
    canvas = size * SUPERSAMPLE
    image = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    def xy(point: tuple[float, float]) -> tuple[float, float]:
        return (point[0] * scale, point[1] * scale)

    draw.rounded_rectangle([0, 0, canvas - 1, canvas - 1], radius=_RADIUS * scale, fill=BACKGROUND)

    axis_width = max(1, round(28 * scale))
    draw.line([xy(p) for p in _AXIS], fill=AXES, width=axis_width, joint="curve")

    curve_width = max(1, round(30 * scale))
    draw.line([xy(p) for p in _bezier(_CURVE)], fill=CURVE, width=curve_width, joint="curve")

    radius = 18.0 * scale
    for dot in _DOTS:
        cx, cy = xy(dot)
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=POINTS)

    return image.resize((size, size), Image.Resampling.LANCZOS)


def write_icons(out_dir: Path) -> list[Path]:
    """Write every icon file into ``out_dir`` and return what was written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    rendered = {size: render(size) for size in PNG_SIZES}
    for size, image in sorted(rendered.items()):
        path = out_dir / f"icon-{size}.png"
        image.save(path)
        written.append(path)
    portable = out_dir / "curvestudio-desktop.png"
    rendered[256].save(portable)
    written.append(portable)
    ico = out_dir / "curvestudio-desktop.ico"
    rendered[256].save(ico, sizes=[(s, s) for s in ICO_SIZES])
    written.append(ico)
    return written


def main(argv: list[str] | None = None) -> int:
    assert __doc__ is not None  # this module always has a docstring
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "out_dir",
        nargs="?",
        default="build/icons",
        type=Path,
        help="directory to write the icon files into (created; default: build/icons)",
    )
    args = parser.parse_args(argv)
    for path in write_icons(Path(args.out_dir)):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
