# Live webcam detection on a Tenstorrent card

    Date      2026-09-23
    Version   v0.1
    Measured  2026-09-24 on a Wormhole n150 L, board 010001851171e0c1.
              Output committed in ../data/

In the spirit of the TT-Studio object detection front end, but built on the
hand-written TTNN YOLOv4 and with no dependencies beyond what the Console
image already carries.

---

## The shape of it

Your camera is on your laptop. The card is in the instance. So the frames have
to travel:

```
  laptop browser                          Console instance
  ──────────────                          ────────────────
  getUserMedia                            server.py
    └─ draw into 320×320 canvas
    └─ toBlob(jpeg, 0.8)
    └─ POST /detect  ──────────────────▶  cv2.imdecode
                                          resize, BGR→RGB
                                          YOLOv4 on the N150
                                          post_processing, NMS
       boxes as JSON  ◀──────────────────  [{x1,y1,x2,y2,conf,cls}]
    └─ draw on overlay canvas
```

Coordinates come back **normalised 0 to 1**, so they scale to whatever size the
video is displayed at. Frames are squashed to a square before sending, which is
exactly what the model's own `load_image` does, so the boxes map straight back
onto the full picture with no aspect correction.

---

## First time, on a fresh instance

Run one command:

```
bash start_webcam_demo.sh
```

It checks the instance, runs the base setup if it has not been done, starts the
server, and prints what to click. First run is 10 to 15 minutes, nearly all of
it downloading the model repository and compiling kernels. After that, under a
minute.

`bash start_webcam_demo.sh --help` for the options. `--skip-setup` goes straight
to the server on a machine you know is prepared. `--port 8080` if 8000 is taken.

### Getting the files onto the instance

Nothing to do. Your instance was prepared with everything already in place,
under `~/challenge`. Start it with:

```bash
bash ~/challenge/webcam_demo/start_webcam_demo.sh
```

If the script says something is missing, the folder has been moved or partly
deleted. `~/challenge/setup/setup_yolov4_on_ttmetal.sh` rebuilds the lot.

### Once it is running

Accept the VS Code port-forwarding prompt, or open the **PORTS** tab beside
TERMINAL and click the forwarded address. Then click **Start camera** and allow
access when the browser asks.

**Keep the browser tab in the foreground while measuring.** Chrome throttles
timers in background tabs, and the JPEG encode then stalls. Measured 14 October
2026: with the tab backgrounded the encode had a median of 4.8 ms and a maximum
of 1825.8 ms, which drags the mean to twelve times the median. Switch to the
terminal while a run is going and your numbers are wrong with nothing to warn
you.

**Chrome will only grant camera access over https or on localhost.** The VS Code
forwarded address is https, so it works. Opening the instance's raw IP over plain
http will not.

### If it does not work

| Symptom | Cause |
|---|---|
| `socket hang up` in the browser | the server is not running, or crashed. Look at the terminal |
| `index.html is missing` page | `index.html` is not in the same folder as `server.py` |
| camera refused, log says so | you opened a plain `http://` address. Use the forwarded https one |
| `STOP: No built tt-metal` | not a Console development-environment image, or the wrong home directory |
| page loads, boxes never appear | check the log panel on the page; the server prints errors there |

---

## Why it is built this way

**Standard library only.** `http.server`, not FastAPI or Flask. Nothing extra to
install, and `HTTPServer` is single-threaded, which serialises access to the
device for free. If you swap in a threading server you must add a lock around
inference, or two frames will hit the card at once.

**The model is resident.** Built once at startup. Rebuilding per request would
cost seconds a frame.

**320×320 rather than 640.** Measured on this card, the confidences agree with
the PyTorch reference at 0.997 at 320 against 0.768 at 640, where the branch's
tiling overrides are rejected much more heavily. So 320 is both quicker and
closer to the reference. Change it and measure what happens.

**One request in flight at a time.** The browser waits for each response before
sending the next frame. Simple, and it stops a slow link from queueing frames
the model will never catch up with.

---

## Reading the numbers on screen

| Panel | What it is |
|---|---|
| Detections | Boxes above the confidence slider, after overlap removal |
| On the machine | Decode plus detection, on the instance. The only part of this a robot would also pay for |
| Queue wait | How long the frame sat behind others already in flight. A property of the browser, not of the card |
| Capture + encode | Grabbing the frame from your camera and encoding it as JPEG, on your laptop |
| Network | The round trip minus everything the server reports, so the camera, the encode, the internet both ways and the drawing |
| Round trip | Browser to browser, the whole journey |
| Frames / s | What you are achieving end to end |
| In flight | How many frames are outstanding at once. Set the control to 1 if you want the timing to mean anything |

**"On the machine" and "Queue wait" are separate panels on purpose.** Add them
together and you get a figure that looks like a hardware cost and is mostly the
browser. That is the single easiest way to quote a wrong number from this demo,
and it is why `bench/` exists.

Measured on this card, the round trip is about **478 ms** against **77 ms** of
work on the machine. Almost everything you see in "round trip" is the network,
the JPEG encode and the browser rather than the hardware. That gap is worth
showing people: it makes the point that the accelerator is rarely the
bottleneck in a real pipeline.

---

## Where its numbers come from

**This has been run, on a Console instance with a Wormhole n150.** It produced
the round-trip measurements committed in `../data/`:
`roundtrip_20260924_094258.jsonl` holds one record per frame as the page
reported it, and `rig_cost.json` summarises two runs from those records.

What it is built from, in case you want to read the originals:

- input preparation from `tests/pcc/test_ttnn_yolov4.py`. It pads the number of
  colour channels from 3 to 16, because the card's memory layout wants data in
  blocks of a fixed width and three does not divide evenly into it
- `YOLOv4Runner` from `runner/runner.py`
- `post_processing` from `post_processing.py`, whose first argument is unused,
  so `None` is passed
- box format `[x1, y1, x2, y2, obj_conf, cls_conf, cls_id]`, with coordinates
  expressed as fractions of the image rather than pixels

**Do not take timings from this page.** The number on screen is dominated by the
journey across the internet to the data centre and back, not by the card. Run
`summarise_roundtrip.py` and it separates the three parts: time on the machine
with the card, time spent queueing, and time on the network. Use `bench/` for
anything you intend to quote.

---

## For the hackathon

This is the glass-box exhibit, and it replaces the TT-Studio dependency in the
challenge sheet, which was never proven on this stack.

Three things to add before it faces a trade-floor audience:

- **A still-image fallback.** Venue wi-fi and camera permissions both fail in
  public. Keep a folder of images the page can cycle through if the camera is
  refused.
- **A visible model and hardware badge.** People will ask what it is running on.
  Put "YOLOv4, Wormhole N150, 320×320" on the page.
- **Reconnect handling.** If the instance expires mid-demo, the page should say
  so plainly rather than silently freezing.

And one caution: the Console instance expires after a day at Developer role. Do
not build a show-floor demo on something that dies overnight without a
documented way to restart it in two minutes.
