# The evaluation harness

This is what scores you at hour 30. You have it from day one so you can build
against it rather than meet it for the first time when it counts.

| file | what it is |
|---|---|
| `submission_template.py` | **Start here.** Copy it, fill in four marked places, and you have a submission. Each place names the file and line numbers to lift from. |
| `submission_api.py` | The interface, plus helpers and checks. Read it when the template refers to `api.something`. |
| `evaluate.py` | Runs your submission over a set of photographs, times it, scores it, writes a result file. |
| `cocomap.py` | The scorer. Needs only numpy. |
| `plot_curve.py` | Puts several result files on a speed-against-accuracy curve and marks which points are on the frontier. Use it on your own runs to see your trade-off. |
| `submission_reference.py` | **The detector exactly as supplied, driven through this interface.** The worked answer to the template: if you are stuck on a TODO, read this. It is the plain path, one frame at a time, which is what you are being asked to beat. |
| `submission_stub.py` | A worked example that needs no card. Run the harness against it to see the whole thing work. |
| `validate_cocomap.py` | The organisers' check that `cocomap.py` agrees with the industry-standard implementation. Needs `pycocotools`, which is not installed here. |

## What you write

Copy the template and fill it in:

```bash
cd ~/challenge
cp harness/submission_template.py my_submission.py
```

It is a class called `Submission` with four methods, and four `TODO` comments
marking what you supply. Each one names the file and the lines in
`bench/bench_stages.py` that already do that step, so none of it is guesswork.
**Only `run` is timed.**

```python
import submission_api as api

class Submission(api.Submission):
    def setup(self, device):   ...   # build the model, load weights. NOT timed.
    def warmup(self, paths):   ...   # a few frames, capture a trace. NOT timed.
    def run(self, paths):      ...   # TIMED. returns {path: [detection, ...]}
    def teardown(self):        ...   # release the card. NOT timed.
```

## Asking for the card you need

The harness opens the card and hands it to `setup`. Two of the settings that
matter most cannot be changed afterwards: `trace_region_size` and
`num_command_queues` are fixed when the device is opened. So if you want to
capture a trace, or overlap the copy to the card with the compute, say so:

```python
class Submission(api.Submission):
    DEVICE_OPTIONS = {"trace_region_size": 6434816, "num_command_queues": 2}
```

The harness reads that before it opens the card and prints what it gave you:

```
  card opened (l1_small_size=10960, num_command_queues=2, trace_region_size=6434816)
```

Leave it empty for the plain path. `l1_small_size` is set for you. 6434816 is
the value `bench/bench_levers.py` uses and is enough for this model at 320x320.

`run` is handed a list of file paths, not decoded images. Unpacking the JPEG and
resizing are therefore inside the timed region, which is deliberate: on the
baseline they account for 2.962 ms of a 69.122 ms frame, and they do not shrink
when the detector does.

Everything expensive belongs in `setup` and `warmup`, which are outside the
timer. Capturing a trace beforehand costs you nothing.

Before calling `setup`, the harness changes the working directory to
`~/tt-metal`, because the supplied model looks for its weights at a relative
path and will not find them otherwise. It changes back afterwards. The paths
you pass on the command line are made absolute first, so this cannot surprise
you. If you load anything yourself, use an absolute path.

## What a detection looks like

```python
{"category_id": 25, "bbox": [93.0, 15.0, 195.0, 278.0], "score": 0.94}
```

- **`category_id`** is COCO's number for the class. It is **not** your model's
  0 to 79 index. Convert with `api.to_coco_id(i)` or the table in
  `../model_index_to_coco_id.json`.
- **`bbox`** is left edge, top edge, width, height, in pixels, in the **original
  photograph's** coordinates. If you shrank the image to run it, scale the boxes
  back with `api.scale_boxes(boxes, (320, 320), (W, H))`.
- **`score`** is between 0 and 1.

Helpers for the common conversions: `api.xyxy_to_xywh`, `api.cxcywh_to_xywh`,
`api.scale_boxes`, `api.to_coco_id`.

## Try it now

Against the day-one sample, with the example submission that needs no card:

```bash
cd ~/challenge
python harness/evaluate.py --selftest \
  --images ~/tt-metal/models/demos/yolov4/resources \
  --annotations sample/instances_sample.json \
  --out /tmp/stub.json
```

Then with your own:

```bash
python harness/evaluate.py \
  --submission my_submission.py \
  --images ~/tt-metal/models/demos/yolov4/resources \
  --annotations sample/instances_sample.json \
  --out my_result.json
```

Or check an output file on its own:

```bash
python harness/submission_api.py --check my_output.json
```

## It tells you when something is wrong

Before scoring, the harness checks your output and prints a `PROBLEM:` line if
it finds one of the mistakes that cost people everything:

- class numbers in 0 to 79 instead of COCO's
- boxes still in the size you ran at rather than the photograph's
- boxes that are not four positive numbers, or scores outside 0 to 1
- nothing returned at all

**A `PROBLEM:` line is more important than the score beside it.** Fix it first.

It also watches the timing. `run` is timed three times and the fastest reading
is reported, because interference can only slow a run down. If the slowest
reading is more than half again the fastest, you get a `DISTURBED:` line: the
machine was busy with something else and all three readings may be inflated.
Close whatever else is running and measure again.

## What you get out

A JSON file and a printed summary: frames per second, milliseconds per frame,
the accuracy score, and a plainer readout of how many real objects you found,
how many detections matched nothing, and how many of those landed on
photographs containing no objects at all. Power is recorded too where the card
reports it.

## Measuring accuracy before hour 30

`~/challenge/practice/` holds 200 photographs with answers. Point the harness at
them and you get a speed and an accuracy for every change you make:

```bash
python harness/evaluate.py --submission my_submission.py \
  --images practice/images --annotations practice/instances.json --out run.json
```

They are not the photographs you are scored on and they do not overlap them.

## At hour 30

The same command, pointed at the photographs you are given and their
annotations file. Nothing about the interface changes. That is the point of
having it now.
