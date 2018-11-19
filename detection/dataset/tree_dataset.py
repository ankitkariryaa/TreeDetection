import os
import numpy as np
import geopandas as gps
import rasterio
from rasterio.transform import rowcol
import matplotlib.pyplot as plt

from detection.dataset.frameinfo import FrameInfo
from detection.utils.logger import logger
from detection.utils.time_utils import time_it

class TreeDataset():
    '''
    Maintains training dataset and implements generators which provide data while training.

    '''
    def __init__(self, base_dir, file_suffix, annotation_file='trees_out.geojson', normalize=False):
        self.base_dir = base_dir
        self.file_suffix = file_suffix
        self.annotation_file = os.path.join(base_dir, annotation_file)
        self.all_frames = self.load_image_data()
        self.dataset_size = len(self.all_frames)
        if normalize:
            self.channel_mean = self.calc_channel_mean()
            self.normalize_frames()

    def load_image_data(self):
        all_files = os.listdir(self.base_dir)
        all_files = [fn for fn in all_files if fn.endswith(self.file_suffix)]
        gdf = gps.read_file(self.annotation_file)

        frame_infos = []
        total_annotations = 0
        for i, fn in enumerate(all_files):
            annotations = []
            image = rasterio.open(os.path.join(self.base_dir, fn))
            for _, row in gdf[gdf.image_idx == fn].iterrows():
                p = row['geometry']
                coord = list(p.coords[0])
                ann = rowcol(image.transform, coord[0], coord[1])
                if not np.isnan(row['KRONE_DM']):
                    tree_size = int(row['KRONE_DM'])
                else:
                    tree_size = 4
                # Convert from metres to pixels and convert to radius from diameter
                tree_size *= 2.5
                tree_size = min(tree_size, 60)
                # Is the annotation correctly initialized???
                annotations.append((ann[0], ann[1], int(tree_size)))
                #print((ann[0], ann[1], tree_size))
            frame_info = FrameInfo(self.base_dir, fn, (0, 0, image.shape[0], image.shape[1]), annotations)
            frame_infos.append(frame_info)
            total_annotations += len(annotations)
        return frame_infos

    def calc_channel_mean(self):
        c1_frames = [frame.img_data[:, :, 0] for frame in self.all_frames]
        c2_frames = [frame.img_data[:, :, 1] for frame in self.all_frames]
        c3_frames = [frame.img_data[:, :, 2] for frame in self.all_frames]
        channel_mean = np.mean(c1_frames), np.mean(c2_frames), np.mean(c3_frames)
        logger.info('Channel mean:{}', channel_mean)
        return channel_mean

    def normalize_frames(self):
        for frame in self.all_frames:
            frame.img_data -= frame.img_data.mean() / (frame.img_data.std() + 1e-8)
        logger.info('Normalized frames with channel mean')

if __name__ == '__main__':
    dataset = TreeDataset('/home/sanjeev/Downloads/subset/', '.jpg', 'trees_out.geojson')
    for f in dataset.all_frames:
        img_out = f.annotated_img()
        plt.imshow(img_out)
        plt.show()
