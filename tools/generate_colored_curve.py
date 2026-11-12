#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generates an ultra-clean trade-off curve with no baseline line and a spacious 2-column legend."""
import json
import pathlib

MILESTONES = [
    ("experiments/runs/01_reference_plain.json", "0. Plain Baseline", "85.8 ms", "#EF4444"),
    ("experiments/runs/02_reference_fast.json", "1. Traced 2CQ", "17.8 ms", "#F97316"),
    ("experiments/runs/03_vectorized_nms.json", "2. Vectorized NMS", "15.8 ms", "#F59E0B"),
    ("experiments/runs/04_vectorized_nms_chunk32.json", "3. Concurrency Tuning", "14.9 ms", "#EAB308"),
    ("experiments/runs/05_vectorized_nms_claude.json", "4. Vectorized Dump", "12.1 ms", "#84CC16"),
    ("experiments/runs/06_vectorized_nms_claude_v2.json", "5. Saliency Gate", "11.5 ms", "#10B981"),
    ("experiments/runs/07_parallel_from_torch.json", "6. Parallel Pre-alloc", "9.8 ms", "#06B6D4"),
    ("experiments/runs/09_double_buffered_overlap.json", "7. Double-Buffered", "7.4 ms", "#3B82F6"),
    ("experiments/runs/10_streaming_pipeline.json", "8. Streaming Pipeline", "5.3 ms", "#8B5CF6"),
    ("experiments/runs/34_gemini.json", "9. Champion (Final fixes)", "3.3 ms / 299 FPS", "#D946EF"),
]

def generate_svg(repo_root: pathlib.Path, out_paths: list[pathlib.Path]):
    pts = []
    for rel_path, label, short_metric, color in MILESTONES:
        p = repo_root / rel_path
        d = json.load(open(p))
        fps = float(d["timing"]["fps"])
        ms = float(d["timing"]["ms_per_frame"])
        mAP = float(d["accuracy"]["mAP_50"])
        pts.append({
            "label": label,
            "metric": short_metric,
            "color": color,
            "fps": fps,
            "ms": ms,
            "mAP": mAP,
        })

    # Dimensions: Clean wide format (860 x 530)
    W, H = 860, 530
    ML, MR, MT, MB = 85, 45, 50, 150
    PW = W - ML - MR
    PH = H - MT - MB

    # Y-axis natural scale: 0.36 to 0.54
    x_min, x_max = 0.0, 320.0
    y_min, y_max = 0.36, 0.54

    def X(fps):
        return ML + (fps - x_min) / (x_max - x_min) * PW

    def Y(mAP):
        return MT + PH - (mAP - y_min) / (y_max - y_min) * PH

    s = []
    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif">')
    s.append('<defs>')
    s.append('  <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">')
    s.append('    <feDropShadow dx="0" dy="1.5" stdDeviation="2" flood-opacity="0.16"/>')
    s.append('  </filter>')
    s.append('  <filter id="glow" x="-30%" y="-30%" width="160%" height="160%">')
    s.append('    <feDropShadow dx="0" dy="0" stdDeviation="4" flood-color="#D946EF" flood-opacity="0.5"/>')
    s.append('  </filter>')
    s.append('</defs>')

    # Background
    s.append(f'<rect width="{W}" height="{H}" fill="#FAFAF9" rx="10"/>')
    s.append(f'<rect x="15" y="15" width="{W-30}" height="{H-30}" fill="#FFFFFF" rx="8" stroke="#E7E5E4" stroke-width="1"/>')

    # Title & Subtitle
    s.append(f'<text x="{ML}" y="38" font-size="15" font-weight="700" fill="#1C1917">Tenstorrent Wormhole n150: Latency vs. Accuracy</text>')
    s.append(f'<text x="{W-MR}" y="38" font-size="11" font-weight="500" fill="#78716C" text-anchor="end">YOLOv4 (320x320) • 200 Test Images</text>')

    # Gridlines Y (mAP) - standard uniform gridlines, no baseline line
    y_ticks = [(0.36, "0.360"), (0.40, "0.400"), (0.44, "0.440"), (0.464, "0.464"), (0.48, "0.480"), (0.52, "0.520")]
    for yt, y_label in y_ticks:
        y_pos = Y(yt)
        s.append(f'<line x1="{ML}" y1="{y_pos:.1f}" x2="{W-MR}" y2="{y_pos:.1f}" stroke="#E7E5E4"/>')
        s.append(f'<text x="{ML-10}" y="{y_pos+4:.1f}" font-size="10.5" font-weight="500" fill="#78716C" text-anchor="end">{y_label}</text>')

    # Gridlines X (FPS)
    x_ticks = [0, 50, 100, 150, 200, 250, 300]
    for xt in x_ticks:
        x_pos = X(xt)
        s.append(f'<line x1="{x_pos:.1f}" y1="{MT}" x2="{x_pos:.1f}" y2="{MT+PH}" stroke="#F5F5F4" stroke-width="1"/>')
        s.append(f'<line x1="{x_pos:.1f}" y1="{MT+PH}" x2="{x_pos:.1f}" y2="{MT+PH+4}" stroke="#78716C" stroke-width="1"/>')
        s.append(f'<text x="{x_pos:.1f}" y="{MT+PH+16}" font-size="11" font-weight="500" fill="#78716C" text-anchor="middle">{xt}</text>')

    # Axes
    s.append(f'<line x1="{ML}" y1="{MT+PH}" x2="{W-MR}" y2="{MT+PH}" stroke="#44403C" stroke-width="1.5"/>')
    s.append(f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT+PH}" stroke="#44403C" stroke-width="1.5"/>')

    # Axis Labels
    s.append(f'<text x="{ML + PW/2}" y="{MT+PH+32}" font-size="11" font-weight="600" fill="#1C1917" text-anchor="middle">Throughput (FPS) → Faster</text>')
    s.append(f'<text x="24" y="{MT + PH/2}" font-size="11" font-weight="600" fill="#1C1917" text-anchor="middle" transform="rotate(-90 24 {MT + PH/2})">Accuracy (mAP@50)</text>')

    # Progression Trail Line
    path_d = []
    for i, p in enumerate(pts):
        cmd = "M" if i == 0 else "L"
        path_d.append(f"{cmd}{X(p['fps']):.1f},{Y(p['mAP']):.1f}")
    s.append(f'<path d="{" ".join(path_d)}" fill="none" stroke="#CBD5E1" stroke-width="1.5" stroke-dasharray="3 3"/>')

    # Plot Dots
    for i, p in enumerate(pts):
        cx, cy = X(p["fps"]), Y(p["mAP"])
        color = p["color"]
        is_champion = "Champion" in p["label"]
        r = 7.5 if is_champion else 5.5

        if is_champion:
            s.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="13" fill="{color}" opacity="0.25" filter="url(#glow)"/>')
            s.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{color}" stroke="#FFFFFF" stroke-width="2" filter="url(#shadow)"/>')
            # Floating callout cleanly above the champion dot
            badge_w, badge_h = 130, 22
            bx = cx - badge_w / 2
            by = cy - 35
            s.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{badge_w}" height="{badge_h}" rx="4" fill="#0F172A" filter="url(#shadow)"/>')
            s.append(f'<text x="{cx:.1f}" y="{by+15:.1f}" font-size="10.5" font-weight="700" fill="#FFFFFF" text-anchor="middle">★ Champion: 299 FPS</text>')
            s.append(f'<line x1="{cx:.1f}" y1="{by+badge_h:.1f}" x2="{cx:.1f}" y2="{cy-r-2:.1f}" stroke="#0F172A" stroke-width="1.5"/>')
        else:
            s.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{color}" stroke="#FFFFFF" stroke-width="1.5" filter="url(#shadow)"/>')

    # ------------------------------------------------------------------------
    # Spacious 2-Column Bottom Legend (Zero Text Overlap Guaranteed)
    # ------------------------------------------------------------------------
    legend_top = MT + PH + 44
    legend_h = 92
    s.append(f'<rect x="{ML}" y="{legend_top}" width="{PW}" height="{legend_h}" fill="#F8FAFC" rx="6" stroke="#E2E8F0" stroke-width="1"/>')

    # 2 columns x 5 rows: Each column has PW / 2 = 365px of width!
    col_w = PW / 2
    row_h = 16.5

    for idx, p in enumerate(pts):
        col = idx // 5
        row = idx % 5
        lx = ML + 16 + col * col_w
        ly = legend_top + 16 + row * row_h

        # Dot badge
        s.append(f'<circle cx="{lx}" cy="{ly}" r="4.5" fill="{p["color"]}" stroke="#FFFFFF" stroke-width="1"/>')

        # Text with 300px+ width available
        is_champ = "Champion" in p["label"]
        name_weight = "800" if is_champ else "600"
        s.append(f'<text x="{lx+10}" y="{ly+3.5}" font-size="10.5" fill="#1E293B">')
        s.append(f'<tspan font-weight="{name_weight}">{p["label"]}</tspan> ')
        s.append(f'<tspan fill="#64748B" font-weight="500">— {p["metric"]}</tspan>')
        s.append('</text>')

    s.append('</svg>')
    svg_content = "\n".join(s)

    for op in out_paths:
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(svg_content)
        print(f"  [success] Written {op}")

if __name__ == "__main__":
    repo = pathlib.Path("SEMICON-Hackathon")
    outs = [
        repo / "experiments/plots/pareto_curve_current.svg",
        repo / "curve.svg",
        pathlib.Path("curve.svg"),
        pathlib.Path("/Users/pazzclanker/.gemini/antigravity/brain/436d13aa-cc6d-4b79-a891-c20f32f700c2/curve.svg")
    ]
    generate_svg(repo, outs)
