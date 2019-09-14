# ====================================================
# @Time    : 2019/9/9 11:36
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : vis_anno.py
# ====================================================

import os.path as osp
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from argparse import ArgumentParser


def draw_bbox(pimage, captions):

    h,w,c = pimage.shape
    src_image = pimage.copy()
    image = pimage.copy()

    pad = 1
    for b in range(len(captions)):
       font = 0.7
       caption = captions[b]
       object_name = caption['name']
       bbox = [int(b) for b in caption['bbox']]
       score = str('%.4f'% caption['score'])
       color= [255,255,255]
       cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, -1)
       bbox_caption = object_name+':'+score
       text_len = len(bbox_caption)*10
       text_color = [0]*3
       if (bbox[1] > 20):
          if bbox[0] + text_len <= w:
              cv2.rectangle(image, (bbox[0]-pad, bbox[1]-18),(bbox[0]+text_len, bbox[1]), color, -1)
              cv2.putText(image, bbox_caption, (bbox[0],bbox[1]-6), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                          font, text_color)
          else:
              cv2.rectangle(image, (bbox[2] - text_len-pad, bbox[1] - 18), (bbox[2], bbox[1]), color, -1)
              cv2.putText(image, bbox_caption, (bbox[2]-text_len, bbox[1] - 6), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                          font, text_color)
       else:
          if bbox[0] + text_len <= w:
              cv2.rectangle(image, (bbox[0]-pad, bbox[1]),(bbox[0]+text_len, bbox[1]+20), color, -1)
              cv2.putText(image, bbox_caption, (bbox[0],bbox[1]+15), cv2.FONT_HERSHEY_COMPLEX_SMALL, font,
                          text_color)
          else:
              cv2.rectangle(image, (bbox[2] - text_len-pad, bbox[1]), (bbox[2], bbox[1] + 20), color, -1)
              cv2.putText(image, bbox_caption, (bbox[2]-text_len, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                          font,text_color)

    cv2.addWeighted(image, 0.8, src_image, 0.2, 0, src_image)

    return src_image


def vis_annotation(image_file, anno_file):
    """
    Visualize annotations
    :param image_file:
    :param anno_file:
    :return:
    """
    anno_tree = ET.parse(anno_file)
    objs = anno_tree.findall('object')
    anno = []
    for idx, obj in enumerate(objs):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)
        anno.append({'name':name, 'score':1, 'bbox':[x1,y1,x2,y2]})

    image = np.asarray(cv2.imread(image_file))
    image = draw_bbox(image, anno)
    image = cv2.resize(image, (1920, 1280))
    cv2.imshow(image_file, image)
    cv2.waitKey(-1)


def main(image_id):
    """
    :param image_id:
    :return:
    """
    image_dir = 'datasets/JPEGImages'
    anno_dir = 'datasets/Annotations'
    image_file = osp.join(image_dir,'{}.jpg'.format(image_id))
    anno_file = osp.join(anno_dir, '{}.xml'.format(image_id))
    assert osp.exists(image_file),'{} not find.'.format(image_file)
    assert osp.exists(anno_file), '{} not find.'.format(anno_file)
    vis_annotation(image_file, anno_file)

if __name__ == "__main__":
    parser = ArgumentParser(description='visualize annotation for image.')
    parser.add_argument('-imageID', dest='imageID', default='042',help='input imageID, e.g., 001')
    args = parser.parse_args()
    main(args.imageID)