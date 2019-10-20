import xml.etree.ElementTree as ET
import multiprocessing as mp
import cv2
import os


def load_samples(cat_name: str, dataset_base: str, n_jobs=4):
    load_n_samples(cat_name, dataset_base, n_jobs)


def load_n_samples(cat_name: str, dataset_base: str, end_after_n=None, n_jobs=4):
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
        files = [l.strip('\n') for l in f.readlines()[:end_after_n]]

    with mp.Pool(n_jobs) as pool:
        args = [(dataset_base, file) for file in files]
        samples = pool.starmap(read_sample, args)

    return samples


def read_sample(dataset_base, file):
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

