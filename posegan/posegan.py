from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras_applications import vgg19

import matplotlib.pyplot as plt
import sys
import numpy as np

from coco_dataset import CocoDataset

class GAN():
    def __init__(self):
        self.input_width = 224
        self.input_height = 224
        self.input_channels = 3

        self.output_size = 1/4
        self.output_width = int(input_width * output_size)
        self.output_height = int(input_height * output_size)

        self.keypoints_amount = len(CocoDataset.filtered_keypoints_list)
        self.bones_amount = len(CocoDataset.filtered_bones_list)

        self.input_shape = (self.input_height, self.input_width, self.input_channels)
        self.output_keypoints_shape = (self.output_height, self.output_width, self.keypoints_amount)
        self.output_pafs_shape = (self.output_height, self.output_width, 2 * self.bones_amount)

        self.total_shape

        self.optimizer = 'sgd'

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.input_shape))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy',
                              optimizer=optimizer)


    def build_generator(self):
        def _upsampling_block(x, index, bn_axis, filters):
            x = UpSampling2D(size=(2, 2), name='upsampling_%s_upsampling' % index)(x)
            x = Conv2D(filters=filters, kernel_size=(3, 3),
                    strides=(1, 1), padding='same', name='upsampling_%s_conv' % index)(x)
            x = BatchNormalization(axis=bn_axis, name='upsampling_%s_bn' % index)(x)
            x = Activation('relu', name='upsampling_%s_relu' % index)(x)
            
            return x

        base_net = vgg19.VGG19(include_top=False, input_shape=self.input_shape, weights=None)
        base_net_output = base_net.get_layer('block5_pool').output
        base_net_output_filters = base_net_output.shape[bn_axis].value
        x = base_net_output

        for i in range(3):
            x = _upsampling_block(
                    x, i+1, bn_axis,
                    int(base_net_output_filters / (2 ** (i+1)))
                )

        keypoints = Conv2D(filters=keypoints_amount, kernel_size=(3, 3),
                strides=(1, 1), padding='same', activation='sigmoid',
                name='keypoints')(x)
        pafs = Conv2D(filters=2 * bones_amount, kernel_size=(3, 3),
                strides=(1, 1), padding='same', activation='tanh',
                name='paf')(x)

        x = Concatenate(name='slice')([keypoints, paf])

        return Model(inputs=base_net.input, outputs=[base_net.input, x])

    def build_discriminator(self):
        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

