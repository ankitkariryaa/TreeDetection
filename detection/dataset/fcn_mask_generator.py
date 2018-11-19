import matplotlib.pyplot as plt

import numpy as np

from detection.dataset.dataset_generator import DatasetGenerator
from detection.dataset.tree_dataset import TreeDataset


class FCNMaskGenerator(DatasetGenerator):
    '''
    Data set generator for generating training and test data for fully connected type networks.
    '''

    def __init__(self, dataset, test_ratio, validation_ratio, horizontal_flip=False):
        super().__init__(dataset, test_ratio, validation_ratio)

    def fcn_data_generator(self, batch_size, patch_size, no_classes, dataset='training', sampling_type='random'):
        while True:
            image_batch = []
            response_maps = []
            idx = self.dataset_idx(dataset)
            while True:
                frame_id = np.random.randint(0, len(idx))
                frame = self.dataset.all_frames[frame_id]
                img_patches = frame.get_random_patches(patch_size, 1)
                for patch in img_patches:
                    img = patch.get_img()
                    ### Remove normalization for the moment
                    # img = (img - img.mean()) / (img.std() + 1e-9)
                    image_batch.append(img)
                # image_batch.extend([img_patch.get_img() for img_patch in img_patches])
                img_masks = [img_patch.ann_mask(no_classes) for img_patch in img_patches]
                response_maps.extend(img_masks)
                if len(image_batch) == batch_size:
                    break
            # logger.info('image batch shape:{}, dataset:{}, batch_size:{}', image_batch.shape, dataset, batch_size)
            yield (np.array(image_batch), np.array(response_maps))

    def fcn_sequential_data_generator(self, batch_size, patch_size, no_classes, dataset='training', sampling_type='random'):
        idx = self.dataset_idx(dataset)
        image_batch = []
        response_maps = []
        for frame_id in idx:
            frame = self.dataset.all_frames[frame_id]
            img_patches = frame.sequential_patches(patch_size, patch_size)# (patch_size[0]//2, patch_size[1]//2))
            print(len(img_patches))
            for patch in img_patches:
                img = patch.get_img()
                image_batch.append(img)
                img_mask = patch.ann_mask2()
                response_maps.append(img_mask)
                if len(image_batch) == batch_size:
                    ib = np.array(image_batch)
                    rm = np.array(response_maps)
                    image_batch = []
                    response_maps = []
                    yield (ib, rm)


if __name__ == '__main__':
    dataset = TreeDataset('/home/sanjeev/Downloads/subset/', '.jpg', 'trees_out.geojson')
    fcn_mask_gen = FCNMaskGenerator(dataset, 0.2, 0.1).fcn_data_generator(1, (224, 224), 1)
    for data in fcn_mask_gen:
        input, output = data
        input_img = input['input_1']
        plt.figure(1), plt.imshow(input_img.squeeze(axis=0))
        out_img = output['class_out']
        plt.figure(2), plt.imshow(np.squeeze(out_img))
        plt.show()
