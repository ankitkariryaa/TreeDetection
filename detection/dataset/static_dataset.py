import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
from detection.dataset.tree_dataset import TreeDataset
from detection.utils.logger import logger


class StaticGenerator():
    '''
    Data set generator for generating training and test data for fully connected type networks.
    '''

    def __init__(self, base_dir, file_prefix, annotation_file_prefix, test_ratio, validation_ratio, horizontal_flip=False):
        self.base_dir = base_dir
        self.file_prefix = file_prefix
        self.annotation_file_prefix = annotation_file_prefix
        files = os.listdir(base_dir)
        self.dataset_size = len(files) // 2
        # self.read_files(file_suffix, annotation_file_suffix)
        self.split_dataset(test_ratio, validation_ratio)


    def filename_to_id(self, filename):
        f = filename.split('.')[0]
        idx = f.split('_')[1]
        return idx

        #Does it work then there are more than 1 or 3 channels
    def open_image(self, filename):
        x = Image.open(os.path.join(self.base_dir, filename))
        im = np.array(x)/255.0
        if (im.ndim == 2):
            im = np.expand_dims(im, -1)
        return im

    def read_files(self):
        all_files = os.listdir(self.base_dir)
        input_img = [fn for fn in all_files if fn.startswith(self.file_prefix)]
        input_img.sort(key=lambda x: self.filename_to_id(x))
        self.input_img = [self.open_image(c) for c in input_img]

        masks = [fn for fn in all_files if fn.startswith(self.annotation_file_prefix)]
        masks.sort(key=lambda x: self.filename_to_id(x))
        self.masks = [self.open_image(c) for c in masks]
        # Asset that they are same and in same order!!
        assert (list(map(self.filename_to_id, input_img)) == list(map(self.filename_to_id, masks)))
        # print(list(map(self.filename_to_id, input_img))[:20])
        # print(list(map(self.filename_to_id, masks))[:20])


    def split_dataset(self, test_ratio, validation_ratio):
        '''
        Splits the dataset into training, testing and validation sets.
        '''
        idx = range(self.dataset_size)
        idx = np.random.permutation(idx)
        training_data_size = int(self.dataset_size * (1 - test_ratio))
        validation_size = int(training_data_size * validation_ratio)
        self.validation_idx = idx[:validation_size]
        self.training_idx = idx[validation_size:training_data_size]
        self.testing_idx = idx[training_data_size:]

        logger.info('Total dataset size:{}, training:{}, validation:{}, test:{}', self.dataset_size,
                    len(self.training_idx), len(self.validation_idx), len(self.testing_idx))

    def static_generator(self, dataset='training'):
        dt = self.dataset_idx(dataset)
        while True:
            idx = np.random.choice(dt, replace=False)
            input_img = [self.open_image('{}_{}.jpg'.format(self.file_prefix, idx))]
            mask = [self.open_image('{}_{}.jpg'.format(self.annotation_file_prefix, idx))]
            yield input_img, mask

    def dataset_idx(self, dataset):
        if dataset == 'testing':
            sample_idx = self.testing_idx
        elif dataset == 'validation':
            sample_idx = self.training_idx
        elif dataset == 'complete':
            sample_idx = range(self.dataset_size)
        else:
            sample_idx = self.training_idx
        return sample_idx

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
