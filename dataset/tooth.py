# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

from torch.utils.data import Dataset

import torch

class SegmentationDataset(Dataset):
    def __init__(self, root, list_path, num_classes,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        
        self.mean = mean
        self.std = std

        self.root = Path(root)
        self.json_path = self.root.joinpath(list_path)
        self.num_classes = num_classes
        self.instances = COCO(self.json_path)
        self.files = self.instances.imgs

        self.class_weights = torch.FloatTensor([1.0, 1.0])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        item = self.instances.loadImgs(index)[0]
        img_path = self.json_path.parents[1].joinpath('images', item['file_name'])
        image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

        img_size = (3, item['height'], item['width'])
        label = np.zeros(img_size, dtype=np.uint8)

        for i in range(1, 4):
            anns = self.instances.loadAnns(self.instances.getAnnIds(imgIds=item['id'], catIds=i))
            for ann in anns:
                label[i-1] += self.instances.annToMask(ann)

        image = self.input_transform(image)
        image = image.transpose((2, 0, 1))
        label[label==255] = 1
        return (image, label, item["file_name"])

    def convert_label(self, label):
        label[label == 0] = self.ignore_label
        label[label < self.ignore_label] = 0
        return label
    
    def input_transform(self, image):
        image = image.astype(np.float32)
        image = image / 255.0
        image -= self.mean
        image /= self.std

        return image
    
    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = preds[i]*(255/2)
            pred = pred.astype(np.uint8)
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, str(name[i]) + '.png'))