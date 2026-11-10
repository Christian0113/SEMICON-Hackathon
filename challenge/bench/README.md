# Where does a frame's time actually go?

    Date      2026-09-23
    Version   v0.1
    Measured  2026-09-24 on a Wormhole n150 L, board 010001851171e0c1.
              Output committed in ../data/

## Why this exists

You will be quoted frame times all weekend, by this repository and by each
other. Most of them will be wrong, and they go wrong in the same three ways.
This tool exists so that at least one number in your change log was taken
properly, and so you can tell when one of yours was not.

| Mistake | What it does to the number |
|---|---|
| The reading is never written to a file | Nothing can be traced back to a machine, a build or a day. Treat it as not measured |
| The stopwatch starts after `cv2.imdecode` | It excludes the JPEG decode that the figure claims to include, which is 2 ms of a frame you are trying to get under 20 |
| The stopwatch spans a queue | If frames are in flight behind a lock, each reading includes waiting for the ones ahead of it. That measures the queue rather than the frame, and inflates the result by an unknown amount |

The third is the one that will catch you. It is also the hardest to notice,
because the number looks stable and plausible while being meaningless.

A frame time read off the webcam demonstration's on-screen panel has all three
problems at once, so do not take a figure from there into your change log. The
panel is for seeing the detector work, not for measuring it.

## What to run

On the instance:

```
cd ~/challenge/bench
bash run_bench.sh
```

About 10 minutes, most of it the first frame compiling kernels. `--quick` runs
60 frames instead of 300 if you only want a sanity check. `--320-only` skips
the second resolution.

## What it measures

No browser, no network, no threads, no lock. One frame at a time, 300 of them,
each stage timed separately with `perf_counter`:

| Stage | What it covers |
|---|---|
| `decode` | `cv2.imdecode` of the JPEG bytes, as the server receives them |
| `prepare` | resize to the model input, BGR to RGB, to torch tensor |
| `upload` | copying the pixels from the ordinary computer into the card's own memory. The code waits for the copy to finish, so the cost is counted here rather than appearing later |
| `model` | `YOLOv4Runner.run`, ending in the read back to host |
| `boxes` | `post_processing`: max-and-argmax, then overlap removal |

Mean, median, p99 and worst for each, plus the total and the implied frame
rate at both the mean and the worst frame. Everything is written to JSON
alongside a `tt-smi` snapshot and the tt-metal commit, so the numbers arrive
with a board, a firmware and a revision attached.

The synchronise after the upload matters. Without it `ttnn` can return before
the transfer has finished and the cost turns up in the `model` row instead.
The script probes for the call, since `ttnn` has moved it between releases, and
prints which one it found. If it finds none, read `upload` and `model` as one
number.

## What it deliberately does not measure

Anything involving the laptop, the browser or the network. In a car the camera
and the chip share a board, so that time does not exist in the product and has
no business in the challenge's headline figure.

## What to do with the result

The `model` row is the plain single-queue path. Compare it against the 5.917 ms
from `test_e2e_performant.py`, which is the same model traced and on two command
queues. The difference is what those changes are worth together.

`run_levers.sh` splits them apart for you, one rung at a time, so you do not
have to guess which of the two earned the milliseconds. Measured on an n150 at
320x320: tracing is effectively the whole of the gap, and the second command
queue is worth nothing at all until frames are handed over as a batch, at which
point it adds about half a millisecond. Run it before you spend a day on the
wrong one.
