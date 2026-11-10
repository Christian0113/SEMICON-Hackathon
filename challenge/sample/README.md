# The day-one sample

Two photographs and twelve boxes, so you can check your submission works long
before it matters.

## What this is for

At hour 30 you are handed 800 photographs you have not seen, and you run one
command to be scored on them. The commonest way to lose that is not a slow
model. It is a submission that returns its boxes in the wrong format, or uses
the wrong category numbers, and produces a score near zero with detections that
were perfectly good.

This sample exists so you find that out on day one instead.

## What it is not

**The score you get on this sample means nothing.** The twelve boxes were placed
by eye, not measured, and two photographs is far too few for any accuracy figure
to be meaningful. Do not quote it, do not optimise against it, and do not read
anything into the number going up or down.

What it does tell you, reliably:

- your code runs under the harness and returns a dict the harness accepts
- your boxes are the right four numbers in the right order, in pixels, in the
  original photograph's coordinates rather than the size you ran at
- your category numbers are COCO's and not your model's own 0 to 79
- your JPEG and PNG decoding both work, since the sample is one of each

## Running it

The two photographs already sit on your machine, where the detector keeps its
own resources. Nothing needs downloading.

```bash
cd ~/challenge
python harness/evaluate.py \
  --submission my_submission.py \
  --images ~/tt-metal/models/demos/yolov4/resources \
  --annotations sample/instances_sample.json \
  --out sample_result.json
```

If the harness prints a `PROBLEM:` line, fix that before anything else. It
catches both of the mistakes described above and tells you which one you made.

You can also check an output file on its own, without running the model:

```bash
python harness/submission_api.py --check my_output.json
```

## Adding your own photographs

Point `--images` at any folder and add entries to a copy of
`instances_sample.json`. The format is COCO's: an `images` list giving each
file's name and size, and an `annotations` list giving `image_id`,
`category_id` and `bbox` as `[left, top, width, height]` in pixels.

## The two photographs

| file | size | what is in it |
|---|---|---|
| `giraffe.jpg` | 320×320 | a giraffe and a zebra, outdoors, heavy foliage |
| `dinning.png` | 640×426 | a living room: television, two people, dining table, three chairs, a potted plant, a vase, a refrigerator |

They are the same two the detector ships with, so you have been looking at them
since the first hour. That is precisely why they are useless for judging
accuracy and fine for checking plumbing.
