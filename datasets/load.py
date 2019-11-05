import xml.etree.ElementTree as ET
import multiprocessing as mp
import numpy as np
import cv2
import os
import math


def load_images(cat_name: str, dataset_base: str, n_jobs=4):
    """
    Expects dataset tree of the following shape
    ├── datasets
    │   ├── Annotations
    │   │   └── +.xml
    │   ├── ImageSets
    │   │   ├── +.txt
    │   ├── JPEGImages
    │   │   ├── +.jpg
    │   └── load.py
    """
    image_set = os.path.join(dataset_base, "ImageSets/" + cat_name + ".txt")
    assert os.path.exists(image_set)

    with open(image_set, "r") as f:
        files = [l.strip("\n") for l in f.readlines()]

    with mp.Pool(n_jobs) as p:
        args = [(dataset_base, file) for file in files]
        return p.starmap(_read_sample, args)


def load_annotations(cat_name: str, dataset_base: str, size=0, n_jobs=4):
    samples = load_images(cat_name, dataset_base, n_jobs)

    classes = dict()
    for sample in samples:
        for cl in sample["classes"]:
            imgs = []
            for bx in cl["bounds"]:
                img = sample["img"]
                if size > 0:
                    dx = bx['x2'] - bx['x1']
                    dy = bx['y2'] - bx['y1']
                    # make region square by extending the smaller dimension
                    if dx > dy:
                        d = (dx - dy) // 2
                        r = img[bx['y1'] - d : bx['y2'] + d, bx['x1'] : bx['x2']]
                        # if impossible shrink the larger one
                        if r.size == 0:
                            r = img[bx['y1'] : bx['y2'], bx['x1'] + d : bx['x2'] - d]
                    else:
                        d = (dy - dx) // 2
                        r = img[bx['y1'] : bx['y2'], bx['x1'] - d : bx['x2'] + d]
                        if r.size == 0:
                            r = img[bx['y1'] + d : bx['y2'] - d, bx['x1'] : bx['x2']]
                    img = cv2.resize(
                        r, dsize=(size, size), interpolation=cv2.INTER_LINEAR
                    )

                imgs.append(img)

            if cl["name"] in classes:
                classes[cl["name"]] += imgs
            else:
                classes[cl["name"]] = imgs

    return classes


def _read_sample(dataset_base, file):
    annotations_path = os.path.join(dataset_base, "Annotations/{}.xml".format(file))
    img_path = os.path.join(dataset_base, "JPEGImages/{}.jpg".format(file))

    sample = dict()
    root = ET.parse(annotations_path).getroot()

    size = root.find("size")
    sample["size"] = (
        int(size.find("width").text),
        int(size.find("height").text),
        int(size.find("depth").text),
    )
    sample["filename"] = root.find("filename").text
    sample["classes"] = []
    sample["img"] = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    for boxes in root.iter("object"):
        name = boxes.find("name").text
        bounds = [
            dict(
                x1=int(box.find("xmin").text),
                y1=int(box.find("ymin").text),
                x2=int(box.find("xmax").text),
                y2=int(box.find("ymax").text),
            )
            for box in boxes.findall("bndbox")
        ]

        # class de-duplication
        names = [c["name"] for c in sample["classes"]]
        if name in names:
            sample["classes"][names.index(name)]["bounds"] += bounds
        else:
            sample["classes"].append({"name": name, "bounds": bounds})
    return sample

def bbx_iou(bbx1, bbx2):
    assert bbx1['x1'] < bbx1['x2']
    assert bbx1['y1'] < bbx1['y2']
    assert bbx2['x1'] < bbx2['x2']
    assert bbx2['y1'] < bbx2['y2']
    
    # determine the coordinates of the intersection rectangle
    x_left = max(bbx1['x1'], bbx2['x1'])
    y_top = max(bbx1['y1'], bbx2['y1'])
    x_right = min(bbx1['x2'], bbx2['x2'])
    y_bottom = min(bbx['y2'], bbx2['y2'])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # compute the area of both AABBs
    bbx1_area = (bbx1['x2'] - bbx1['x1']) * (bbx1['y2'] - bbx1['y1'])
    bbx2_area = (bbx2['x2'] - bbx2['x1']) * (bbx2['y2'] - bbx2['y1'])
    
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bbx1_area + bbx2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou