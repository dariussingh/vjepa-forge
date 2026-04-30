from __future__ import annotations

from pathlib import Path

import numpy as np


def _polyline_points(values: np.ndarray, width: int, height: int) -> str:
    if len(values) == 1:
        return f"0,{height // 2}"
    xs = np.linspace(0, width, num=len(values))
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax == vmin:
        ys = np.full_like(xs, height / 2.0)
    else:
        ys = height - ((values - vmin) / (vmax - vmin)) * (height - 20) - 10
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys))


def write_timeline_svg(
    output_path: str | Path,
    title: str,
    scores: np.ndarray,
    labels: np.ndarray,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width = 900
    height = 260
    label_rects: list[str] = []
    if len(labels) > 0:
        xs = np.linspace(0, width, num=len(labels), endpoint=False)
        rect_width = max(width / max(len(labels), 1), 1.0)
        for idx, value in enumerate(labels):
            if int(value) > 0:
                label_rects.append(
                    f'<rect x="{xs[idx]:.2f}" y="0" width="{rect_width:.2f}" height="{height}" '
                    'fill="#ffd6d6" opacity="0.65" />'
                )
    polyline = _polyline_points(scores, width, height)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height + 40}">
<rect width="100%" height="100%" fill="#ffffff" />
<text x="16" y="24" font-family="monospace" font-size="18" fill="#222222">{title}</text>
<g transform="translate(0,40)">
{''.join(label_rects)}
<line x1="0" y1="{height - 10}" x2="{width}" y2="{height - 10}" stroke="#999999" stroke-width="1" />
<polyline fill="none" stroke="#0055aa" stroke-width="2" points="{polyline}" />
</g>
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")
