import cv2
from keras import models, layers
from detection.utils.image_utils import local_maxima, get_annotated_img
import matplotlib.pyplot as plt



def upsample_conv(filters, kernel_size, strides, padding):
    return layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)


def upsample_simple(filters, kernel_size, strides, padding):
    return layers.UpSampling2D(strides)


def UNet(input_shape, upsample_mode='SIMPLE', net_scaling=None, gaussian_noise=0.1, edge_crop=16, weight_file = None):
        if upsample_mode == 'DECONV':
            upsample = upsample_conv
        else:
            upsample = upsample_simple
        input_img = layers.Input(input_shape[1:], name='RGB_Input')
        pp_in_layer = input_img
        if net_scaling is not None:
            pp_in_layer = layers.AvgPool2D(net_scaling)(pp_in_layer)

        # pp_in_layer = layers.GaussianNoise(gaussian_noise)(pp_in_layer)
        # pp_in_layer = layers.BatchNormalization()(pp_in_layer)

        c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(pp_in_layer)
        c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(c1)
        n1 = layers.BatchNormalization()(c1)
        p1 = layers.MaxPooling2D((2, 2))(n1)

        c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(p1)
        c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c2)
        n2 = layers.BatchNormalization()(c2)
        p2 = layers.MaxPooling2D((2, 2))(n2)

        c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p2)
        c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c3)
        n3 = layers.BatchNormalization()(c3)
        p3 = layers.MaxPooling2D((2, 2))(n3)

        c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p3)
        c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c4)
        n4 = layers.BatchNormalization()(c4)
        p4 = layers.MaxPooling2D(pool_size=(2, 2))(n4)

        c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p4)
        c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c5)

        u6 = layers.UpSampling2D((2, 2))(c5)
        n6 = layers.BatchNormalization()(u6)
        u6 = layers.concatenate([n6, n4])
        c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
        c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c6)

        u7 = layers.UpSampling2D((2, 2))(c6)
        n7 = layers.BatchNormalization()(u7)
        u7 = layers.concatenate([n7, n3])
        c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u7)
        c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c7)

        u8 = layers.UpSampling2D((2, 2))(c7)
        n8 = layers.BatchNormalization()(u8)
        u8 = layers.concatenate([n8, n2])
        c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u8)
        c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c8)

        u9 = layers.UpSampling2D((2, 2))(c8)
        n9 = layers.BatchNormalization()(u9)
        u9 = layers.concatenate([n9, n1], axis=3)
        c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(u9)
        c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(c9)

        d = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
        # d = layers.Cropping2D((edge_crop,edge_crop))(d)
        # d = layers.ZeroPadding2D((edge_crop, edge_crop))(d)
        if net_scaling is not None:
            d = layers.UpSampling2D(net_scaling)(d)

        seg_model = models.Model(inputs=[input_img], outputs=[d])
        if weight_file:
            seg_model.load_weights(weight_file)
        seg_model.summary()
        return seg_model


if __name__ == '__main__':
    batch_size = 1
    detector = UNet([batch_size, 252, 252, 3], 1e-3, 1, weight_file='../../weights/unet.hdf5')
