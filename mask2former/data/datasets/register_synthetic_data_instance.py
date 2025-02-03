# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass


# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/datasets/builtin.py
# Modified by Jitesh Jain (https://github.com/praeclarumjj3)
# ------------------------------------------------------------------------------


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os
import json
from detectron2.data.datasets.builtin_meta import  _get_builtin_metadata
from detectron2.data.datasets.coco import register_coco_instances



_PREDEFINED_SPLITS_COCO = {
    "synthetic_data_instance_train": ("synthetic_data/train", "synthetic_data/annotations/panoptic2instances_train.json"),
    # "synthetic_data_instance_val": ("coco/val2017", "coco/annotations/panoptic2instances_val2017.json"),
}

def get_metadata(categories):
    meta = {}

    thing_classes = [k["name"] for k in categories if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in categories]

    meta["thing_classes"] = thing_classes
    meta["stuff_classes"] = stuff_classes

    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(categories):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id
    return meta



def register_panoptic2instances_coco(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_COCO.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_builtin_metadata("coco"), 
            # metadata,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


_root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets"))
# categories = json.load(open(os.path.join(_root, "synthetic_data/synthetic_categories.json")))
# metadata = get_metadata(categories)
register_panoptic2instances_coco(_root)