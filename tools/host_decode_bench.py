"""Host-side decode/resize benchmark. Runs on any laptop, no card needed.

Mirrors submission_reference_fast._decode (cv2.imread -> resize 320 -> BGR2RGB)
and compares it with JPEG reduced-size decoding, which lets libjpeg skip work
when the photo is at least 2x (or 4x) the target on both sides.

    python tools/host_decode_bench.py challenge/practice/images --repeats 3
"""
import argparse, glob, os, statistics as st, time
import cv2, numpy as np

RES = (320, 320)
cv2.setNumThreads(1)          # one decode per thread, like the worker pool

def ref(path):
    f = cv2.imread(path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(cv2.resize(f, RES), cv2.COLOR_BGR2RGB)

def reduced(path, w, h):
    s = min(w // RES[0], h // RES[1])
    flag = cv2.IMREAD_REDUCED_COLOR_4 if s >= 4 else cv2.IMREAD_REDUCED_COLOR_2 if s >= 2 else cv2.IMREAD_COLOR
    f = cv2.imread(path, flag)
    return cv2.cvtColor(cv2.resize(f, RES), cv2.COLOR_BGR2RGB)

def time_it(fn, paths, sizes, repeats):
    best = None
    for _ in range(repeats):
        ts = []
        for p in paths:
            t0 = time.perf_counter(); fn(p, *sizes[p]) if fn is reduced else fn(p); ts.append((time.perf_counter() - t0) * 1e3)
        if best is None or sum(ts) < sum(best): best = ts
    return best

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("images"); ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    paths = sorted(p for p in glob.glob(os.path.join(a.images, "*")) if p.lower().endswith((".jpg", ".jpeg", ".png")))
    sizes = {}
    for p in paths:
        h, w = cv2.imread(p).shape[:2]; sizes[p] = (w, h)
    print(f"{len(paths)} photos; mean {st.mean(w*h for w,h in sizes.values())/1e3:.0f} kpx; "
          f"{sum(min(w//320,h//320)>=2 for w,h in sizes.values())} are >=2x the target on both sides")
    for p in paths[:5]: ref(p)                       # warm the file cache
    r = time_it(ref, paths, sizes, a.repeats); q = time_it(reduced, paths, sizes, a.repeats)
    diffs = [np.abs(ref(p).astype(np.int16) - reduced(p, *sizes[p]).astype(np.int16)).mean() for p in paths]
    q95 = lambda x: sorted(x)[int(.95 * len(x))]
    print(f"{'':24}{'mean':>8}{'median':>8}{'p95':>8}  ms per photo, one thread")
    print(f"{'reference decode+resize':24}{st.mean(r):8.3f}{st.median(r):8.3f}{q95(r):8.3f}")
    print(f"{'reduced decode+resize':24}{st.mean(q):8.3f}{st.median(q):8.3f}{q95(q):8.3f}")
    print(f"saving {st.mean(r)-st.mean(q):.3f} ms/photo ({(1-st.mean(q)/st.mean(r))*100:.0f}%); "
          f"mean abs pixel difference at 320x320: {st.mean(diffs):.2f} / 255 (max photo {max(diffs):.2f})")
    print("Laptop numbers. Re-run on the instance before quoting; check mAP with evaluate.py.")

if __name__ == "__main__":
    main()
