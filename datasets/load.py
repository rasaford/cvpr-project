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
    image_set = os.path.join(dataset_base, 'ImageSets/' + cat_name + '.txt')
    assert os.path.exists(image_set)

    with open(image_set, 'r') as f:
        files = [l.strip('\n') for l in f.readlines()]

    with mp.Pool(n_jobs) as p:
        args = [(dataset_base, file) for file in files]
        return p.starmap(_read_sample, args)


def load_annotations(cat_name: str, dataset_base: str, size=0, n_jobs=4):
    samples = load_images(cat_name, dataset_base, n_jobs)

    classes = dict()
    for sample in samples:
        for cl in sample['classes']:
            imgs = [sample['img'][bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    for bbox in cl['bounds']]
            if cl['name'] in classes:
                classes[cl['name']] += imgs
            else:
                classes[cl['name']] = imgs

    if size > 0:
        resampled = dict()
        with mp.Pool(n_jobs) as p:
            for name, imgs in classes.items():
                args = [(img, size) for img in imgs]
                resampled[name] = p.starmap(_resize_sample, args)
        classes = resampled

    return classes


def _resize_sample(img, size):
    y, x = img.shape[:2]
    if x > y:
        d = (x - y) // 2
        img = img[:, d: y + d]
    else:
        d = (y - x) // 2
        img = img[d: x + d, :]

    return cv2.resize(img, dsize=(size, size))


def _read_sample(dataset_base, file):
    annotations_path = os.path.join(dataset_base,
                                    'Annotations/{}.xml'.format(file))
    img_path = os.path.join(dataset_base, 'JPEGImages/{}.jpg'.format(file))

    sample = dict()
    root = ET.parse(annotations_path).getroot()

    size = root.find('size')
    sample['size'] = (int(size.find('width').text),
                      int(size.find('height').text),
                      int(size.find('depth').text))
    sample['filename'] = root.find('filename').text
    sample['classes'] = []
    sample['img'] = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    for boxes in root.iter('object'):
        name = boxes.find('name').text
        bounds = [[
            int(box.find('xmin').text),
            int(box.find('ymin').text),
            int(box.find('xmax').text),
            int(box.find('ymax').text)]
            for box in boxes.findall('bndbox')
        ]

        # class de-duplication
        names = [c['name'] for c in sample['classes']]
        if name in names:
            sample['classes'][names.index(name)]['bounds'] += bounds
        else:
            sample['classes'].append({
                'name': name,
                'bounds': bounds
            })
    return sample
