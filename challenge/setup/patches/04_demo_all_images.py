#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Patch models/demos/yolov4/demo.py: run the image demo over every image.

  Date      2026-11-05
  Version   v1.0

FIXES
    The plain image demo (test_yolov4) runs with batch size 1 and takes only
    image_files[0], where image_files comes from os.listdir(). That order is
    whatever the filesystem returns, not alphabetical. On the 24 Sep reference
    instance the first file was giraffe.jpg; on the first event instance,
    5 Nov 2026, it was dinning.png and the giraffe was never processed. A team
    that drops ten photographs into resources/ would get one output back, and
    which one would depend on the machine.

WHAT CHANGES
    run_yolov4() now sorts the image list and runs the compiled pipeline once
    per batch until every image has been through it. Each image is written to
    yolov4_predictions/ as before, under the same file name pattern. A short
    batch at the end is padded with its own last image and only the real
    images are saved.

    Each image's detection count is logged, and an image with none is logged
    as a warning. The test still reports PASSED either way, so look at the log
    line or the output image, not the exit code.

    Model, weights, resolution, conf_thresh (0.3) and nms_thresh (0.4) are
    untouched. Pipeline set-up and compile still happen once per resolution.

ORDER
    Apply after 02_demo_imports.py. If 02 is re-run it restores demo.py from
    demo.py.orig, which removes this patch; re-run this one afterwards, as the
    setup script does. Idempotent: a second run on a patched file does nothing.

Usage:
    python3 04_demo_all_images.py <tt-metal-root>
    e.g.  python3 04_demo_all_images.py ~/tt-metal
"""

import sys
from pathlib import Path

MARKER = "# --- hackathon demo patch 04"

START = "    # Load and preprocess images\n    image_files = [\n"
END = "        plot_boxes_cv2(orig_images[i], boxes[i], output_path, class_names)\n"

BLOCK = """    # --- hackathon demo patch 04: every image in the folder, in sorted order ---
    image_files = sorted(
        os.path.join(input_loc, f) for f in os.listdir(input_loc) if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    assert image_files, f"No .png, .jpg or .jpeg images in {input_loc}"
    logger.info(f"{len(image_files)} images in {input_loc}, {batch_size} per batch")

    conf_thresh = 0.3
    nms_thresh = 0.4
    class_names = load_coco_class_names()
    output_dir = "yolov4_predictions"
    os.makedirs(output_dir, exist_ok=True)

    for start in range(0, len(image_files), batch_size):
        chunk = image_files[start : start + batch_size]
        n_real = len(chunk)
        chunk = chunk + [chunk[-1]] * (batch_size - n_real)

        torch_images = []
        orig_images = []
        for image_path in chunk:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Image not found or unreadable: {image_path}")
            orig_images.append(image)
            torch_images.append(preprocess_image(image, resolution).unsqueeze(0))

        torch_input_tensor = torch.cat(torch_images, dim=0)
        tt_inputs_host, _ = test_infra._setup_l1_sharded_input(device, torch_input_tensor)
        outputs = pipeline.enqueue([tt_inputs_host]).pop_all()
        result_boxes, result_confs = get_model_result(outputs[0], resolution, mesh_composer=output_mesh_composer)
        boxes = post_processing(torch_images, conf_thresh, nms_thresh, [result_boxes, result_confs])

        for i in range(n_real):
            name = os.path.basename(chunk[i])
            img_base = os.path.splitext(name)[0]
            output_path = os.path.join(output_dir, f"ttnn_yolov4_{img_base}_{resolution[0]}x{resolution[1]}.jpg")
            plot_boxes_cv2(orig_images[i], boxes[i], output_path, class_names)
            if len(boxes[i]) == 0:
                logger.warning(f"{name}: no detections above conf {conf_thresh}")
            else:
                logger.info(f"{name}: {len(boxes[i])} detections -> {output_path}")
"""


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        print(f"error: give the tt-metal root, e.g. python3 {sys.argv[0]} ~/tt-metal")
        return 2

    target = Path(sys.argv[1]).expanduser() / "models" / "demos" / "yolov4" / "demo.py"
    if not target.is_file():
        print(f"error: not found: {target}")
        return 1

    text = target.read_text()
    if MARKER in text:
        print(f"already patched: {target}")
        return 0

    if text.count(START) != 1:
        print(f"error: start anchor found {text.count(START)} times, expected 1.")
        print("demo.py differs from the commit this patch was written against.")
        return 1
    s = text.index(START)
    e = text.find(END, s)
    if e < 0:
        print("error: end anchor not found after the start anchor.")
        return 1
    e += len(END)

    target.write_text(text[:s] + BLOCK + text[e:])
    print(f"patched: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
