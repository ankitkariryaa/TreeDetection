import matplotlib.pyplot as plt
import os

import cv2
import numpy as np
import geopandas as gp
import rasterio
from sklearn.decomposition import PCA
from scipy.stats import describe
from detection.utils import image_utils
from detection.utils.image_utils import get_annotated_img, filename_to_id


np.random.seed(43)


class FrameInfo():
    '''
    Contains annotated image information such as image data, annotations etc.
    '''

    def __init__(self, base_dir, img_id, roi, annotations):
        self.base_dir = base_dir
        self.img_id = img_id
        # add buffer to region of interest ...
        # roi: (x1,y1,x2,y2)
        # We dont need bbox
        self.roi = roi#roi[0] - bbox[0], roi[1] - bbox[0], roi[2] + bbox[1], roi[3] + bbox[1]
        image = rasterio.open(os.path.join(self.base_dir, self.img_id))
        read_image = image.read()
        self.full_img = np.transpose(read_image, (1, 2, 0)).astype(np.uint8)
        self.img_data = self.full_img
        self.annotations = annotations
        self.annotations_mask = self.annotated_img(mask_only=True)
        self.all_seq_patches = []

    def annotated_img(self, mask_only=True, shape='circle', size=-1):
        annotated_img = get_annotated_img(self.img_data, self.annotations, mask_only, shape, size)
        return annotated_img

    def get_random_patches(self, patch_size, no_of_patches):
        '''
        Randomly samples no_of_patches of patch_size from image.
        '''
        img_shape = self.img_data.shape
        x = np.random.randint(0, img_shape[0] - patch_size[0], no_of_patches)
        y = np.random.randint(0, img_shape[1] - patch_size[1], no_of_patches)
        xy = zip(x, y)
        img_patches = []
        for i, j in xy:
            img_patch = Patch(self, j, i, patch_size)
            img_patches.append(img_patch)
        return img_patches

    def sequential_patches(self, patch_size, step_size):
        '''
        Returns all sequential patches from image separated by step.
        '''
        if len(self.all_seq_patches) == 0:
            img_shape = self.img_data.shape
            x = range(0, img_shape[0] - patch_size[0], step_size[0])
            y = range(0, img_shape[1] - patch_size[1], step_size[1])
            xy = [(i, j) for i in x for j in y]
            img_patches = []
            for i, j in xy:
                img_patch = Patch(self, j, i, patch_size)
                img_patches.append(img_patch)
            self.all_seq_patches = img_patches
        return self.all_seq_patches


class Patch(object):
    '''
    Represents a patch inside an input image.
    '''

    def __init__(self, frame_info, startx, starty, patch_size):
        self.frame_info = frame_info
        self.startx = startx
        self.starty = starty
        self.patch_size = patch_size
        self.mask_type = 'circle'  # 'rectangle' 'gaussian'
        self.__find_annotations()

    def get_img(self):
        img_data = self.frame_info.img_data
        img_patch = img_data[self.startx:self.startx + self.patch_size[1], self.starty:self.starty + self.patch_size[0]]
        return img_patch

    def __find_annotations(self):
        '''
        Finds annotations whose bounding box completely lie in the patch.
        '''
        annotations = []
        for ann in self.frame_info.annotations:
            row, col, size = ann
            row_start, col_start, row_end, col_end = row - size, col - size, row + size, col + size
            if self.startx < row_start and self.starty < col_start and self.startx + self.patch_size[0] > row_end and self.starty + self.patch_size[1] > col_end:
                annotations.append(ann)
        self.ann_relative = annotations
        self.annotations = [(ann[0] - self.startx, ann[1] - self.starty, ann[2]) for ann in annotations]

    def annotated_img(self):
        ann_patch = get_annotated_img(self.get_img(), self.annotations)
        return ann_patch

    def ann_mask(self, no_classes):
        img_mask = np.zeros(self.patch_size + (no_classes,))
        print('Annotations in patch:',len(self.annotations))
        for ann in self.annotations:
            x, y, s = ann  # s here is size and not the number of classes. Add support for multiple classes in future.
            mask_max = 255
            mask_size = -1  # -1 means fill completely
            if self.mask_type == 'circle':
                cv2.circle(img_mask, (y, x), ann[2], mask_max, mask_size)  # x,y are inverted in cv2
            elif self.mask_type == 'rectangle':
                bbox_size = (int(ann[2]), int(ann[2]))
                cv2.rectangle(img_mask, (y - bbox_size[0], x - bbox_size[0]), (y + bbox_size[1], x + bbox_size[1]),
                              mask_max, mask_size)  # x,y are inverted in cv2
            else:  # 'gaussian'
                i = 0  # s if no_classes > 1 else 0
                gaussian_k = image_utils.gaussian_kernel((s*2+1, s*2+1), 8, mask_max)
                sx = max(0, x - s)
                sy = max(0, y - s)
                ex = min(self.patch_size[0] - 1, x + s)
                ey = min(self.patch_size[1] - 1, y + s)
                m = np.maximum(img_mask[sx:ex + 1, sy:ey + 1, i], gaussian_k)
                img_mask[sx:ex + 1, sy:ey + 1, i] = m
        return img_mask

    def ann_mask2(self):
        img_data = self.frame_info.annotations_mask
        img_patch = img_data[self.startx:self.startx + self.patch_size[1], self.starty:self.starty + self.patch_size[0]]
        return img_patch