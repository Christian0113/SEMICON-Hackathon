# Make a camera see fast enough to act on

The Chip Architecture and Integration challenge at a student hackathon in
Munich, 10 to 12 November 2026.

Everything Tenstorrent supplies for it is in this repository.

A car has to spot a pedestrian in time to brake. A delivery robot has to spot
the same pedestrian, but it walks, so it has far longer to decide. Every
machine that moves points a camera at the world, runs a program that says
what is in the picture, and acts on the answer. This repository is that loop,
running on a Tenstorrent card, with the tools to measure it honestly.

## The number to beat

One frame, measured on the card you will be given:

| | ms | |
|---|---|---|
| unpack the photo | 2.135 | turn the compressed JPEG back into pixels |
| resize and convert | 0.827 | shrink to 320x320, reorder the colours |
| hand it to the card | 2.903 | copy the pixels into the card's memory |
| run the detector | 61.062 | YOLOv4, on the card |
| work out the boxes | 2.195 | likeliest object per box, drop the overlaps |
| **whole frame** | **69.122** | one stopwatch, not a sum of the rows |
| frames a second | 14.5 | at the average |
| worst frame of 300 | 83.203 | on a quiet machine; read the note below before optimising for it |

Mean of 300 frames on `giraffe.jpg` at 320x320. Wormhole n150 L, board
`010001851171e0c1`, firmware bundle 19.11.0, KMD 2.9.0, tt-metal
`9f9cd4fd590`, 24 September 2026. Raw output in [`data/`](data/), and you can
reproduce all of it with `bash bench/run_bench.sh`.

**The worst frame measures the machine as much as it measures your code.** A
single slow frame is what a robot trips over, so tail latency is a real concern
in the field. On a shared cloud instance it is not a number you can act on. The
same 300-frame run repeated on 26 October 2026, on a host busy with other
tenants' work, gave a worst frame of 130.777 ms against a median of 74.214 ms;
the 83.203 ms above came from a quieter machine on a different day. Your scored
speed is the fastest of three runs for exactly this reason. Work from the
median, and read the worst frame as a report on how busy the machine was.

**The table above and the scoring tool measure different things, and will not
agree.** The table is one 320x320 photograph decoded from memory, 300 times.
The scoring tool reads a folder of real photographs from disk, at fifteen
different sizes. The most common is 640x480, at 23% of them, and across the
whole set they average about 2.6 times the pixels of a 320x320 frame, which is
that much more work to unpack and resize. The unmodified detector measures near
69 ms in the table and near 79 ms through the scoring tool, and both numbers are
correct. Compare like with like: your improvement is measured against your own
earlier run of the same tool, never against the other one.

**The size of the photograph on disk is not the size the detector runs at.**
Every submission decodes the file and resizes it before the model sees anything,
so a 640x480 photograph and a 500x375 one both arrive at the detector as whatever
resolution you chose to run. The file size costs you decode and resize time on
the host; it has nothing to do with the model resolution discussed under
`run_levers.sh` below.

**Expect a few percent of spread between cards.** A second n150 L, board
`01000186119060ee`, measured 66.0 ms for the same work on 14 October 2026.
Nothing in the software changed. Measure your own machine before you compare
yourself against anything, including this table.

**Under 15 milliseconds a frame is achievable on this card with no measurable
loss of accuracy.** Two techniques get you most of the way. A **trace** is a
recording of the whole sequence of instructions the card needs for one frame,
made once. After that the host says "run the recording" instead of issuing
instructions one at a time, and the card stops waiting to be told what to do
next. A **second command queue** gives you a second channel to the card, so the
next photograph is copied across while the current one is still being
processed, instead of the two taking turns. Both are documented in the model's
own README at the pinned commit, which lands on your instance during setup, and our harnesses for it are in
[`bench/`](bench/). Reading about it takes five minutes. Getting a working
15 ms on your own machine takes rather longer, and the harder part is what is
left afterwards: the four host-side rows in the table above come to 8.060 ms,
they do not shrink when the detector does, and neither this repository nor
tt-metal does anything about them.

## Quick start

On your instance:

```bash
# everything is already installed at ~/challenge, so just start the demo
bash ~/challenge/webcam_demo/start_webcam_demo.sh
```

Then accept the VSCode port-forwarding prompt, or open the PORTS tab and
click the forwarded address. It has to be the forwarded https address,
because a browser will not hand your camera to a plain http page.

To measure rather than watch:

```bash
bash ~/challenge/bench/run_bench.sh
```

## Everything we give you, file by file

Four folders, forty-one files. These are the ones you will actually touch.

### `setup/` — how the detector got onto your machine

Your instance arrives with this already done. It is here so you can see exactly
what was done to your machine, and so you can rebuild if you break something.

| file | what it is |
|---|---|
| `setup_yolov4_on_ttmetal.sh` | Fetches the detector at a fixed version, downloads the trained weights, applies the four patches below, and runs the model once to fill the kernel cache. About ten minutes. |
| `patches/01_builder_activations.py` | Compatibility fix. The detector was written in May 2026 and needs small adjustments for today's software. |
| `patches/02_demo_imports.py` | The same, for the demo's imports. |
| `patches/03_perblock_pcc_tests.py` | Repairs the per-stage accuracy tests, which did not run at all on a single card. Without it you can see that your answer got worse but not which of the seven stages it started in. |
| `patches/04_demo_all_images.py` | Makes the demo run every image in `resources/`, in name order. Before it, the demo ran only the first file the folder happened to list. |

### `bench/` — measurement. Start here and come back after every change.

| file | what it is | when to run it |
|---|---|---|
| `run_bench.sh` | **Where the time goes.** One photograph at a time, split into the five steps: unpack, resize, hand to the card, run the detector, work out the boxes. Throws away the first frames, then times the whole frame with a separate stopwatch as well as the parts. Writes a timestamped folder. | First, to get your own machine's baseline. Then after every change. |
| `run_loop.sh` | **The fast path.** Measures the model running as a continuous pipeline rather than one frame at a time, at 320 with one channel to the card, 320 with two, and 640 with one. | Once you have a baseline and want the large win. |
| `run_levers.sh` | **What each change is worth on its own.** A ladder of four rungs, each adding exactly one thing, so the gap between two rungs is what that one thing bought. | When you need to say which change earned which milliseconds, which is what the jury will ask. |
| `bench_stages.py`, `bench_loop.py`, `bench_levers.py` | The programs the three scripts above drive. Read them. They are commented and they document what went wrong while they were written. | |
| `README.md` | Why an early figure of "about 70 ms" was wrong and must not be used. | Worth five minutes. |

**All three check the machine before they start** and stop with a red `STOP:`
if the detector is missing, if tt-metal is not built, or if a file has been
moved. They also record which card and which software version produced the
numbers, into `board.txt` and `env.txt`, because a timing with no machine
attached cannot be compared with anything.

**`run_loop.sh` additionally proves itself before it measures.** Step 3 hands
the model two clearly different photographs and checks that different data
actually reaches the card. If it does not, the model is processing the same
picture over and over and any speed from it would be meaningless, so the script
stops and says `UNSOUND` rather than print a fast, useless number. The first
three attempts at writing that tool failed exactly here and only noticed at the
end, which is why the check now comes first.

**`run_levers.sh` at 320 reproduces the published figure.** Run on a Wormhole
n150 on 2026-10-26, `trace_2cq` came out at 5.448 ms a frame against the
published 5.917 ms, and every configuration matched the reference output.

**640x640 does not currently work on the supplied model.** In the same run the
plain path raised `TT_FATAL` and stopped, and all three traced configurations
ran but failed their numerical check at PCC 0.768, where 320 reaches 0.996. The
same 0.7678787216744429 appears in `data/loop_results_20260924_093510/loop_640_1cq.json`
from 24 September, so it is reproducible and not a bad afternoon. Raising
resolution is therefore not a lever you can use yet. If you find the cause we
would genuinely like to know.

### `webcam_demo/` — a live demonstration for your pitch

| file | what it is |
|---|---|
| `start_webcam_demo.sh` | Starts it. Then accept the port-forwarding prompt. |
| `server.py` | Keeps the model loaded on the card and answers frames from the page. |
| `index.html` | The page. Captures your laptop camera and draws the boxes. |
| `summarise_roundtrip.py` | Separates the three parts of the delay you see on screen: time on the machine with the card, time spent queueing, and time on the network. |
| `README.md` | How it is built, and why its on-screen figure is mostly network time. |

The number this shows is dominated by the journey to the data centre and back.
Use `bench/` for anything you intend to quote.

### `data/` — the evidence behind every number we claim

| file | what it is |
|---|---|
| `bench_320.json` | The per-stage timings behind the table at the top of this README. |
| `board.txt`, `env.txt` | Which card, which firmware, which software version produced them. |
| `loop_results_20260924_093510/` | Nine files. Our own measurement of the fast pipeline at 320×320, which is what shows the target is reachable. The 640×640 file in there is kept for a different reason: its `validated` field reads `FAILED: 0.7678787216744429`, which is the evidence that 640 does not currently produce correct output. Do not read it as a speed you can aim at. |
| `rig_cost.json`, `roundtrip_*.jsonl` | What the demonstration rig itself costs, so the network delay can be subtracted rather than guessed at. |
| `README.md` | What each file contains and which edits were made for public release. |

Diff your own runs against ours. If we are wrong, we would rather you found out.

### `harness/` — the code that will score you, which you have from the start

| file | what it is |
|---|---|
| `submission_template.py` | Copy this and fill in four marked places. Each one names the file and lines in `bench/bench_stages.py` that already do that step. |
| `submission_api.py` | The interface, the COCO id table and the box helpers. |
| `evaluate.py` | Runs your submission over a set of photographs, times it, scores it, writes a result file. This is what we run at hour 30. |
| `cocomap.py` | The scorer. Needs only numpy, so it behaves identically on every machine. |
| `plot_curve.py` | Draws your speed-against-accuracy curve from several result files. Each run of `evaluate.py` is one point; several points are a curve, and the curve is what the jury looks at. |
| `submission_reference.py` | The detector as supplied, written through the interface. The worked answer to the template, and the baseline your result is compared against. |
| `submission_stub.py` | A worked example that needs no card, so you can see the whole thing run before you have written anything. |
| `validate_cocomap.py` | Our check that the scorer agrees with the industry-standard implementation. Needs `pycocotools`, which is not installed here. |
| `README.md` | What to write, what a detection looks like, and how to run it. |

Before scoring, it checks your output and prints a `PROBLEM:` line if it finds
one of the mistakes that cost people everything: class numbers in 0 to 79
instead of COCO's, boxes still in the size you ran at, boxes that are not four
positive numbers, or nothing returned at all. **A `PROBLEM:` line matters more
than the score beside it.**

### `sample/` — check your submission works on day one

| file | what it is |
|---|---|
| `instances_sample.json` | Two photographs and twelve boxes, in the same format as the hour-30 material. |
| `README.md` | How to run it and, more importantly, what it does and does not tell you. |

The commonest way to lose at hour 30 is not a slow model. It is a submission
that returns its boxes in the wrong order, or in the size you ran at rather than
the photograph's, or with your model's own class numbers instead of COCO's. Any
of those gives a score near zero with detections that were perfectly good. Run
this on day one and you find out then instead.

**The score on this sample means nothing.** Twelve boxes placed by eye across
two photographs you have been staring at since the first hour. It tells you your
plumbing is right, nothing more.

### In the root

| file | what it is |
|---|---|
| `model_index_to_coco_id.json` | Eighty numbers. Position in your model's own class list, converted to COCO's category number. Only needed if you bring a different model, and then it is the difference between a good score and nearly zero. See the section on changing the model. |

### The rest

`LICENSE`, `LICENSE_understanding.txt`, `NOTICE`, `CODE_OF_CONDUCT.md`,
`CONTRIBUTING.md`, `SECURITY.md`, `.gitignore` and a GitHub workflow. Licensing,
the list of third-party components, and the contribution and conduct rules.

The detector itself is not in this repository. It lives in tt-metal, pinned
to a commit so every instance is identical:
[`models/demos/yolov4` at `3a0e398`](https://github.com/tenstorrent/tt-metal/tree/3a0e3987db56711450ec04d388dadcd723d80c98/models/demos/yolov4).
The detector is not one calculation. It is seven stages in a row:
`downsample1` through `downsample5`, which repeatedly shrink the picture while
building up a richer description of it, then `neck`, which recombines those
descriptions so that small objects are not lost along with the detail, then
`head`, which turns the result into boxes and categories. Each stage feeds the
next, so a mistake early on quietly spoils everything after it.

That directory gives you each stage in its own file, so you can change one at a
time; a plain PyTorch version that runs on an ordinary processor and acts as the
known-good answer; and tests that compare against it.

The whole-model test tells you whether your answer got worse:

```bash
cd ~/tt-metal
python -m pytest -x -q \
  "models/demos/yolov4/tests/pcc/test_ttnn_yolov4.py::test_yolov4[0-pretrained_weight_true-0]"
```

The per-stage tests tell you **which** stage it started going wrong in, which is
the difference between an afternoon of guessing and a ten-minute fix:

```bash
cd ~/tt-metal
python -m pytest -q --tb=line \
  models/demos/yolov4/tests/pcc/test_ttnn_downsample{1,2,3,4,5}.py \
  models/demos/yolov4/tests/pcc/test_ttnn_neck.py \
  models/demos/yolov4/tests/pcc/test_ttnn_head.py
```

Run them after every change you make, not at the end.

Two of these fail on the detector as supplied, before you have touched anything:
`downsample4` reports a poor correlation and `downsample5` raises an error.
Those are real problems in the model we were given, not something you broke.
Note them and carry on.

## Take care when comparing figures

Nothing is held back for later: `bench/` above holds every harness we used,
including the two behind the 15 ms figure. The model's README quotes 166 FPS at
320x320, which is 6.02 ms of device-side end-to-end time. The 69.122 ms in
the table above is a whole frame on one stopwatch with the host stages
included. Those measure different things, and if you quote one while
measuring the other somebody will ask you about it.

## Two numbers you choose: confidence and overlap

The detector does not hand back a tidy list. It produces thousands of candidate
boxes, each with a score, and two thresholds turn that into an answer. You pick
both, nobody picks them for you, and both move your result.

**Confidence threshold.** Every candidate carries a number for how sure the
model is. Below the threshold, the box is thrown away. Raise it and you report
fewer things that are not there, but you lose faint, small and distant objects.
Lower it and you catch more of those, at the cost of some invention.

**Overlap threshold.** The model usually finds the same object several times,
with slightly different boxes. The clean-up step keeps the highest-scoring box
and deletes any other box overlapping it by more than this fraction. Raise it
and near-duplicates survive. Lower it and you merge hard, which can delete a
second object that genuinely overlaps the first: two people standing close
together become one person.

Three different pairs are already in play in the code you have been given,
which is exactly why yours has to be stated:

| where | confidence | overlap |
|---|---|---|
| the baseline run behind this README, in `data/bench_320.json` | 0.3 | 0.4 |
| tt-metal's own `postprocess`, which the demo uses | 0.25 | 0.7 |
| `evaluate.py`, for the readable summary beside your score | 0.25 | 0.5 |

### Set the confidence low for scoring. We measured what it is worth.

The score walks every detection you return in score order and credits the
correct ones, so a box you cut away can never be credited. Cutting with a high
confidence threshold therefore **lowers** your score rather than raising it.

The supplied detector, over the 200 practice photographs, on a Wormhole n150,
26 October 2026:

| confidence | mAP | ms a frame |
|---|---|---|
| 0.3 | 0.3762 | 78.863 |
| 0.001 | **0.4639** | 82.708 |

Same detector, same photographs, one line changed. At 0.3 roughly two thirds of
the real objects had no box submitted for them at all. The extra cost is about
4 ms, all of it in overlap removal, and it is worth paying.

`submission_template.py` starts you at 0.001 for this reason. **Raise it for the
camera demonstration**, where nothing is scored and a page covered in faint
boxes looks terrible. Those are two different jobs and they want two different
settings.

So the two thresholds are a lever like any other, and like the others they cost
something. Measure, do not assume.

## What you hand in

Five things. Have them ready before hour 30 rather than after.

| | |
|---|---|
| **Your submission file** | The one you made from `harness/submission_template.py`. It holds the single configuration you are claiming, and it is what we run against the 800 photographs. |
| **Your change log**, one page | What you changed, in order, and what each change was worth. Speed from `bench/run_bench.sh`, accuracy from the scoring tool against `practice/`. Say which numbers you measured and which you estimated. |
| **Your settings** | Resolution, which model, confidence threshold, overlap threshold. |
| **Your starting measurement** | From `bench/run_bench.sh` before you changed anything. Cards differ by a few percent, so your improvement is measured against your own. |
| **Your curve** | The result files from your practice runs, and the SVG from `plot_curve.py`. This is the evidence behind the change log. |

For the pitch, something that runs rather than slides: the camera demo at the
speed you are claiming. Rehearse it once, and have a recording in case the
room's network is against you.

## At hour 30 you are given 800 photographs you have not seen

Everything above is measured on two photographs, a giraffe and a dining room.
Two is not enough to know whether your faster version still works. Thirty hours
in, you receive 800 photographs. You have not seen them, you cannot tune
against them, and we are not telling you where they come from until afterwards.
Your final score comes from those.

They are already on your machine, locked. We read out the password on the day
and you run:

```bash
bash ~/challenge/unlock_hour30.sh <the password>
```

That opens them into `~/challenge/testset/`, checks nothing was damaged on the
way, and prints the command that scores you. Nothing is downloaded at the venue,
so the connection cannot let you down.

### How you are scored

Two numbers, both measured by us on your machine, using the same code for every
team.

**Accuracy.** The standard detection score, called mAP. It covers whether you
found the right objects, in the right places, without reporting things that were
not there. Higher is better. Alongside it you will see a plainer breakdown: how
many real objects you found, how many detections matched nothing, and how many
of those landed on the 48 photographs that contain no objects at all.

**Speed.** Frames per second across the same 800 photographs, timed from being
handed the list of files to producing the detections. Unpacking the JPEG,
resizing and working out the boxes are all inside the timer. Setting up and
warming up are outside it, so capturing a trace beforehand costs you nothing.

The run is timed three times and the fastest reading is the one that counts.
Other work on the host or the card can only add time to a run, never remove it,
so repeated measurements of identical work scatter upwards from a floor, and the
floor is the honest figure. Three consecutive runs of the unmodified detector on
2026-10-26 gave 88.352, 87.579 and 99.790 ms a frame for byte-identical output,
so a single reading can be 14% pessimistic. Use `--repeats` if you want a
different number while you are experimenting; the judges use three.

Sometimes it is much worse than 14%. A run over the 200 practice photographs the
same day produced a slowest reading 6.1 times the fastest, with the host-side
suppression step inflated by the same factor, which says something else had the
CPU rather than anything the card was doing. When the readings disagree by more
than half, the harness prints a `DISTURBED:` line telling you so. The fastest
reading is still the honest one, but when you see that line all three may be
inflated, so close your camera demo and your other terminals and measure again
before you write the number in your change log.

Neither number is worth anything on its own. A fast version that misses half the
objects scores badly, and so does an accurate version no quicker than the one
you started with. You are judged on the trade-off you chose and on how well you
can justify it.

### Do this in the first hour, whatever you plan to do

Two mistakes produce a score near zero from detections that were perfectly good.
Both are invisible until something checks for them, and both take minutes to fix
once you know. So check on day one rather than at hour 30:

```bash
cd ~/challenge
python harness/evaluate.py --selftest \
  --images ~/tt-metal/models/demos/yolov4/resources \
  --annotations sample/instances_sample.json --out /tmp/check.json
```

That runs a worked example that needs no card, so it works before you have
written anything. Swap `--selftest` for `--submission your_file.py` once you
have something of your own. Ignore the accuracy number it prints, which is
meaningless on two photographs. Look only for lines beginning `PROBLEM:`.

**`PROBLEM: ... class number in 0..79`** means your detections carry your
model's own class index instead of COCO's category number. The two are the same
order but different numbers. Convert with `api.to_coco_id(i)`, or the table in
`model_index_to_coco_id.json`. This hits any team that brings a different model.

**`PROBLEM: your boxes reach 20% of the frame where the real objects reach
70%`** means your boxes are still in the size you ran the model at rather than
the original photograph's. This hits any team that changes resolution, which is
most of you. Scale them back with `api.scale_boxes(boxes, (320, 320), (W, H))`.

No `PROBLEM:` line means your output is the right shape, and from then on you
can concentrate on speed.

**You already have the tool that will score you.** It is `harness/`, and it is
the same code we will run at hour 30. Build against it from the first hour, and
check it works using `sample/`. Only the photographs are held back.

## Measuring accuracy while you work: the practice photographs

You cannot improve what you cannot measure, and until hour 30 you have no way
of knowing whether a change cost you accuracy. So you are given a separate set
to work against.

`~/challenge/practice/` holds **200 photographs with their answers recorded**.
They are not the photographs you are scored on, they do not overlap them, and
you may tune against them as hard as you like. That is what they are for.

The same tool that scores you at hour 30 scores you against these, so every
change you make can produce a speed and an accuracy together:

```bash
cd ~/challenge
python harness/evaluate.py \
  --submission my_submission.py \
  --images practice/images \
  --annotations practice/instances.json \
  --out run_$(date +%H%M).json
```

Run that after every change worth keeping. Each run writes a result file, and
each result file is one point: a speed and an accuracy. Collect several and draw
your trade-off:

```bash
python harness/plot_curve.py run_*.json --out my_curve.svg
```

That curve is what your change log should be built from, and it is what makes
the difference between "we think trace was worth about 40 ms" and knowing.

Two cautions. Two hundred photographs is enough to tell a real change from
noise, but a difference of a few thousandths means nothing, so do not chase it.
And the practice set is small enough to tune into, particularly on thresholds:
a setting that is perfect here and nowhere else will show up at hour 30.

## You may change anything, including the model

Replacing the detector with a different one is allowed and some of you will try
it. Three things to know before you do.

Downloading a newer model is the easy part. Making it run well on this hardware
is the whole difficulty, and it is what the detector you were given has already
solved for you. Budget a day.

You lose two things that only make sense for the detector you were given: the
per-stage accuracy tests, which compare each of its seven stages against a
known-good version, and `run_levers.sh`, which switches that detector's
particular optimisations on and off one at a time to show what each is worth.
With a different model you are back to measuring whole runs and reasoning about
the difference.

**Your model's class numbers will be wrong.** Almost every detector emits a
plain 0 to 79. COCO's category numbers are not 0 to 79, though they are in the
same order, so submitting yours unconverted scores close to zero with perfectly
good detections. Convert with `model_index_to_coco_id.json`, and confirm it
worked by running the day-one check above. You will see the problem in seconds
instead of at hour 30.

**Have something that runs end to end by hour 24.** This is the part teams
underestimate. A new model that nearly works at hour 30 scores the same as no
model at all. If you are not through the day-one check by hour 24, go back to
the detector you were given and spend the remaining time making it fast.

## Two traps that have already cost us a day each

**The card may be working on a different photograph than you think.** One of
the functions you are given takes the photograph you pass it, then ignores it
and uses the one it already had. There is no error and no warning. You get a
sensible-looking set of boxes for the wrong photograph, and every number you
measure afterwards is meaningless.

To check: push two clearly different photographs through your pipeline and
confirm you get two different sets of boxes. Do this before you trust any
measurement, and again after any change to how you feed the model.

**Measure the work, not the waiting.** If photographs arrive faster than the
card can process them, they line up and wait their turn. A stopwatch started
when a photograph arrives therefore measures how long it queued as well as how
long the work took. Our own first figure, 70 ms, was wrong for this reason: we
read it off the camera demo's screen, where each frame had up to three others
waiting ahead of it.

Use `bench/run_bench.sh` for anything you intend to quote. It handles one
photograph at a time with nothing queued behind it, throws away the warm-up,
and writes the result to a file you can show someone.

## Help

Documentation at [docs.tenstorrent.com](https://docs.tenstorrent.com),
community at [discord.gg/tenstorrent](https://discord.gg/tenstorrent), and
three Tenstorrent engineers on site.

## Contributing

Bugs via [GitHub Issues](../../issues), changes via pull requests, reviewed
weekly. During the event we are not merging. Keep your work in your own copy on
your instance. See [CONTRIBUTING.md](CONTRIBUTING.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md). Security problems go through
[SECURITY.md](SECURITY.md) rather than the issue tracker.

## License

Tenstorrent code here is licensed Apache-2.0, matching tt-metal, which the
patches in `setup/` modify. See [LICENSE](LICENSE), and
[LICENSE_understanding.txt](LICENSE_understanding.txt) for Tenstorrent's
clarification of how it applies. [NOTICE](NOTICE) lists third-party material
and where each piece comes from.

The detector, tt-metal and the YOLOv4 weights are not redistributed here;
`setup/` fetches each from its own source. YOLOv4 is not an Ultralytics
model.
