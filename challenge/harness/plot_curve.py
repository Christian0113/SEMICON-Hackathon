# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Put result files on a speed-against-accuracy curve.

Every run of evaluate.py writes one result file, which is one point: a speed
and an accuracy. Run it at several settings and you have a curve, which says
far more about what you understand than any single point does.

    python harness/plot_curve.py result_*.json --out curve.svg

With several teams' files it also marks who is on the frontier, meaning nobody
beat them on both speed and accuracy at once.

No libraries beyond the standard one, so it runs anywhere.
"""
import argparse, json, pathlib, sys


def load(paths):
    pts = []
    for p in paths:
        try:
            d = json.load(open(p))
        except Exception as e:
            print(f"  skipping {p}: {e}", file=sys.stderr); continue
        fps = (d.get("timing") or {}).get("fps")
        mAP = (d.get("accuracy") or {}).get("mAP_50")
        if fps is None or mAP is None:
            print(f"  skipping {p}: no timing/accuracy", file=sys.stderr); continue
        pts.append({"label": d.get("team") or pathlib.Path(p).stem,
                    "fps": float(fps), "mAP": float(mAP),
                    "ms": (d.get("timing") or {}).get("ms_per_frame"),
                    "watts": (d.get("power") or {}).get("watts_mean"),
                    "images": d.get("images"), "file": str(p)})
    return pts


def frontier(pts):
    """Points nobody beats on both axes at once."""
    out = []
    for a in pts:
        if not any(b is not a and b["fps"] >= a["fps"] and b["mAP"] >= a["mAP"]
                   and (b["fps"] > a["fps"] or b["mAP"] > a["mAP"]) for b in pts):
            out.append(a)
    return sorted(out, key=lambda p: p["fps"])


def svg(pts, front, path, baseline=None):
    W, H, M = 760, 460, 70
    xs = [p["fps"] for p in pts] or [0, 1]
    ys = [p["mAP"] for p in pts] or [0, 1]
    x0, x1 = 0, max(xs) * 1.15 or 1
    y0, y1 = max(0, min(ys) - 0.1), min(1.0, max(ys) + 0.08) or 1
    if y1 <= y0: y1 = y0 + 0.1
    def X(v): return M + (v - x0) / (x1 - x0) * (W - M - 24)
    def Y(v): return H - M - (v - y0) / (y1 - y0) * (H - M - 36)

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="Helvetica,Arial,sans-serif">',
         f'<rect width="{W}" height="{H}" fill="#fbfaf8"/>']
    for i in range(6):                                   # grid
        gy = M + i * (H - M - 36) / 5
        s.append(f'<line x1="{M}" y1="{gy:.1f}" x2="{W-24}" y2="{gy:.1f}" stroke="#e3e1dc"/>')
    s.append(f'<line x1="{M}" y1="{H-M}" x2="{W-24}" y2="{H-M}" stroke="#1a1a1c" stroke-width="1.5"/>')
    s.append(f'<line x1="{M}" y1="{M}" x2="{M}" y2="{H-M}" stroke="#1a1a1c" stroke-width="1.5"/>')
    for i in range(6):
        v = y0 + i * (y1 - y0) / 5
        s.append(f'<text x="{M-10}" y="{Y(v)+4:.1f}" font-size="11" fill="#5c5c66" '
                 f'text-anchor="end">{v:.2f}</text>')
    for i in range(6):
        v = x0 + i * (x1 - x0) / 5
        s.append(f'<text x="{X(v):.1f}" y="{H-M+18}" font-size="11" fill="#5c5c66" '
                 f'text-anchor="middle">{v:.0f}</text>')
    s.append(f'<text x="{(W+M)/2:.0f}" y="{H-18}" font-size="13" fill="#1a1a1c" '
             f'text-anchor="middle">frames a second  (faster to the right)</text>')
    s.append(f'<text x="18" y="{H/2:.0f}" font-size="13" fill="#1a1a1c" text-anchor="middle" '
             f'transform="rotate(-90 18 {H/2:.0f})">accuracy, mAP  (better upwards)</text>')

    if len(front) > 1:                                   # the frontier itself
        pth = " ".join(f"{'M' if i==0 else 'L'}{X(p['fps']):.1f},{Y(p['mAP']):.1f}"
                       for i, p in enumerate(front))
        s.append(f'<path d="{pth}" fill="none" stroke="#7c3f00" stroke-width="2" '
                 f'stroke-dasharray="6 4" opacity="0.65"/>')
    fset = {id(p) for p in front}
    placed = []                      # keep labels off one another
    for p in sorted(pts, key=lambda q: (q["fps"], q["mAP"])):
        on = id(p) in fset
        isbase = baseline is not None and p is baseline
        fill = "#1a1a1c" if isbase else ("#7c3f00" if on else "#a2a2ad")
        r = 7 if isbase else (6 if on else 4.5)
        cx, cy = X(p["fps"]), Y(p["mAP"])
        s.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}"/>')
        ly = cy + 4
        while any(abs(ly - q) < 13 and abs(cx - qx) < 120 for qx, q in placed):
            ly -= 14
        placed.append((cx, ly))
        if ly != cy + 4:             # show which point the label belongs to
            s.append(f'<line x1="{cx+r+1:.1f}" y1="{cy:.1f}" x2="{cx+9:.1f}" y2="{ly-4:.1f}" '
                     f'stroke="#a2a2ad" stroke-width="0.8"/>')
        s.append(f'<text x="{cx+10:.1f}" y="{ly:.1f}" font-size="12" '
                 f'fill="#1a1a1c">{p["label"]}</text>')
    if baseline is not None:
        s.append(f'<line x1="{M}" y1="{Y(baseline["mAP"]):.1f}" x2="{W-24}" '
                 f'y2="{Y(baseline["mAP"]):.1f}" stroke="#1a1a1c" stroke-width="1" '
                 f'stroke-dasharray="3 5" opacity="0.5"/>')
        s.append(f'<text x="{W-26}" y="{Y(baseline["mAP"])-8:.1f}" font-size="11" '
                 f'fill="#5c5c66" text-anchor="end">accuracy of the detector as supplied</text>')
    s.append("</svg>")
    pathlib.Path(path).write_text("\n".join(s))


def main():
    a = argparse.ArgumentParser()
    a.add_argument("results", nargs="+")
    a.add_argument("--out", default="curve.svg")
    a.add_argument("--baseline", help="the result file for the untouched detector")
    args = a.parse_args()

    pts = load(args.results)
    if not pts: sys.exit("no usable result files")
    base = None
    if args.baseline:
        base = next((p for p in pts if p["file"].endswith(args.baseline)), None)
        if base is None:
            bl = load([args.baseline])
            if bl: base = bl[0]; pts.append(base)

    n = {p["images"] for p in pts if p["images"]}
    if len(n) > 1:
        print(f"  WARNING: these results are over different numbers of photographs {sorted(n)}. "
              f"They are not comparable.")

    front = frontier(pts)
    fset = {id(p) for p in front}
    print(f"\n  {'':2} {'label':<16}{'fps':>8}{'ms/frame':>10}{'mAP':>8}{'watts':>8}   ")
    for p in sorted(pts, key=lambda q: -q["fps"]):
        mark = "*" if id(p) in fset else " "
        base_mark = "<- as supplied" if base is not None and p is base else ""
        w = f"{p['watts']:.1f}" if p["watts"] else "  -"
        ms = f"{p['ms']:.2f}" if p["ms"] else "   -"
        print(f"  {mark:2} {p['label']:<16}{p['fps']:>8.1f}{ms:>10}{p['mAP']:>8.4f}{w:>8}   {base_mark}")
    print(f"\n  * on the frontier: nobody beat them on speed and accuracy at once")
    if base is not None:
        better = [p for p in pts if p is not base and p["fps"] > base["fps"] and p["mAP"] >= base["mAP"]]
        print(f"  faster than the supplied detector with no accuracy lost: "
              f"{', '.join(p['label'] for p in better) if better else 'none'}")
    svg(pts, front, args.out, base)
    print(f"  written {args.out}")


if __name__ == "__main__":
    main()
