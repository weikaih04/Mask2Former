#!/usr/bin/env python
# ------------------------------------------------------------------------------
# Reference: https://github.com/cocodataset/panopticapi/blob/master/converters/panoptic2detection_coco_format.py
# Modified by Jitesh Jain, Updated to Polygon format
# ------------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import argparse
import numpy as np
import json
import time
import multiprocessing
import cv2  # Added OpenCV for contour extraction
import PIL.Image as Image
from panopticapi.utils import get_traceback, rgb2id, save_json

@get_traceback
def convert_panoptic_to_detection_coco_format_single_core(
    proc_id, annotations_set, categories, segmentations_folder, things_only
):
    annotations_detection = []
    for working_idx, annotation in enumerate(annotations_set):
        if working_idx % 100 == 0:
            print(f'Core: {proc_id}, {working_idx} from {len(annotations_set)} images processed')

        file_name = f"{annotation['file_name'].rsplit('.')[0]}.png"
        try:
            pan_format = np.array(Image.open(os.path.join(segmentations_folder, file_name)), dtype=np.uint32)
        except IOError:
            raise KeyError(f'No prediction PNG file for id: {annotation["image_id"]}')
        
        pan = rgb2id(pan_format)

        for segm_info in annotation['segments_info']:
            if things_only and categories[segm_info['category_id']]['isthing'] != 1:
                continue

            mask = (pan == segm_info['id']).astype(np.uint8)
            
            # Extract contours (polygon format)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygons = []
            for contour in contours:
                if len(contour) > 4:  # Ensure it's a valid polygon
                    polygons.append(contour.flatten().tolist())  # Convert to list format
            
            if not polygons:  # If no valid polygons, skip
                continue
            
            segm_info.pop('id')
            segm_info['image_id'] = annotation['image_id']
            segm_info['segmentation'] = polygons  # Store as polygon format
            annotations_detection.append(segm_info)

    print(f'Core: {proc_id}, all {len(annotations_set)} images processed')
    return annotations_detection


def convert_panoptic_to_detection_coco_format(input_json_file, segmentations_folder, output_json_file, categories_json_file, things_only):
    start_time = time.time()

    if segmentations_folder is None:
        segmentations_folder = input_json_file.rsplit('.', 1)[0]

    print("CONVERTING...")
    print("COCO panoptic format:")
    print(f"\tSegmentation folder: {segmentations_folder}")
    print(f"\tJSON file: {input_json_file}")
    print("TO")
    print("COCO detection format")
    print(f"\tJSON file: {output_json_file}")
    if things_only:
        print("Saving only segments of things classes.")
    print('\n')

    print(f"Reading annotation information from {input_json_file}")
    with open(input_json_file, 'r') as f:
        d_coco = json.load(f)
    annotations_panoptic = d_coco['annotations']

    with open(categories_json_file, 'r') as f:
        categories_list = json.load(f)
    categories = {category['id']: category for category in categories_list}

    cpu_num = multiprocessing.cpu_count()
    annotations_split = np.array_split(annotations_panoptic, cpu_num)
    print(f"Number of cores: {cpu_num}, images per core: {len(annotations_split[0])}")
    
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, annotations_set in enumerate(annotations_split):
        p = workers.apply_async(
            convert_panoptic_to_detection_coco_format_single_core,
            (proc_id, annotations_set, categories, segmentations_folder, things_only)
        )
        processes.append(p)

    annotations_coco_detection = []
    for p in processes:
        annotations_coco_detection.extend(p.get())

    for idx, ann in enumerate(annotations_coco_detection):
        ann['id'] = idx

    d_coco['annotations'] = annotations_coco_detection
    categories_coco_detection = []
    for category in d_coco['categories']:
        if things_only and category['isthing'] != 1:
            continue
        category.pop('isthing')
        categories_coco_detection.append(category)
    
    d_coco['categories'] = categories_coco_detection
    save_json(d_coco, output_json_file)

    t_delta = time.time() - start_time
    print(f"Time elapsed: {t_delta:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The script converts panoptic COCO format to detection COCO format using polygon masks."
    )
    parser.add_argument('--things_only', action='store_true', help="discard stuff classes")
    args = parser.parse_args()
    
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    root = os.path.join(_root, "synthetic_data")
    input_json_file = os.path.join(root, "annotations", "panoptic_train.json")
    output_json_file = os.path.join(root, "annotations", "panoptic2instances_train.json")
    categories_json_file = os.path.join(root, "panoptic_coco_categories.json")
    segmentations_folder = os.path.join(root, "panoptic_train")
    
    convert_panoptic_to_detection_coco_format(input_json_file,
                                              segmentations_folder,
                                              output_json_file,
                                              categories_json_file,
                                              args.things_only)