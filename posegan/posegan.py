from __future__ import print_function, division

from keras import backend as K
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras_applications import vgg19

import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import cv2
import math

from coco_dataset import CocoDataset

class GAN():
    def __init__(self):
        self.coco = CocoDataset(
            mode='val',
            year='2017',
            dataset_dir='C:\\COCO-Dataset'
        )
        self.images_dir = os.path.join(self.coco.dataset_dir, 'val2017')

        self.input_width = 224
        self.input_height = 224
        self.input_channels = 3

        self.output_size = 1/4
        self.output_width = int(self.input_width * self.output_size)
        self.output_height = int(self.input_height * self.output_size)

        self.keypoints_amount = len(self.coco.filtered_keypoints_list)
        self.bones_amount = len(self.coco.filtered_bones_list)

        self.input_shape = (self.input_height, self.input_width, self.input_channels)
        self.output_shape = (self.output_height, self.output_width, self.keypoints_amount + 2 * self.bones_amount)

        self.optimizer = 'sgd'

        '''
        Build and compile the discriminator
        '''
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=self.optimizer,
                                   metrics=['accuracy'])

        '''
        Build the generator
        '''
        self.generator = self.build_generator()

        '''
        The generator takes noise as input and generates imgs
        '''
        z = Input(shape=(self.input_shape))
        img = self.generator(z)

        '''
        For the combined model we will only train the generator
        '''
        self.discriminator.trainable = False

        '''
        The discriminator takes generated images as input and
        determines validity
        '''
        validity = self.discriminator(img)

        '''
        The combined model (stacked generator and discriminator)
        Trains the generator to fool the discriminator
        '''
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy',
                              optimizer=optimizer)


    def get_input_image(self, image_filepath):
        def get_rgb_image(image):
            if isinstance(image, str):
                image = cv2.imread(image)
            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return rgb_image

        '''
        read image from filepath:
        '''
        rgb_image = get_rgb_image(image_filepath)
        resized_image = cv2.resize(rgb_image, (self.input_width, self.input_height))
        preprocessed_image = preprocess_input(
            resized_image.astype(float), mode='tf'
        )
        
        return preprocessed_image


    def get_heatmaps(self, image_id):
        variance = 1
        '''
        get keypoints list of the image, from the annotation file.
        '''
        annotation_list = self.coco.get_annotation_list_by_image_id(image_id)
        keypoints = self.coco.get_keypoints(annotation_list)
        
        '''
        get original image dimensions.
        '''
        image_w, image_h = self.coco.get_width_height_by_image_id(image_id)

        '''
        initialize set of empty heatmaps (for each keypoint + background).
        '''
        heatmap = np.zeros((self.output_height, self.output_width, self.keypoints_amount))
        
        gaussian_threshold = 0.05
        max_gaussian_distance = (
            -((variance**2) * math.log(gaussian_threshold))
        ) ** 0.5
        variance_squared = variance ** 2

        window_size = 2 * math.ceil(max_gaussian_distance) + 1
        half_window_size = math.floor(window_size / 2)
        window = np.zeros((window_size, window_size))
        for wx in range(window_size):
            for wy in range(window_size):
                real_wx = wx - half_window_size
                real_wy = wy - half_window_size
                window[wy, wx] = math.exp(
                    -((np.linalg.norm((real_wx, real_wy))) ** 2 / variance_squared)
                )

        for i in range(len(self.coco.filtered_keypoints_list)):
            keypoint_type = self.coco.filtered_keypoints_list[i]
            curr_keypoints = []
            for (p, keypoint) in list(enumerate(keypoints[keypoint_type])):
                if keypoint is None:
                    continue
                x, y = keypoint
                x *= self.output_width / image_w
                y *= self.output_height / image_h
                x = min(self.output_width - 1, math.floor(x))
                y = min(self.output_height - 1, math.floor(y))
                curr_keypoints.append((x, y))
            if curr_keypoints == []:
                continue

            for (kx, ky) in curr_keypoints:
                for wx in range(window_size):
                    for wy in range(window_size):
                        real_wx = wx - half_window_size
                        real_wy = wy - half_window_size
                        curr_pixel_update_x = kx + real_wx
                        curr_pixel_update_y = ky + real_wy
                        curr_pixel_update_x = min(
                            self.output_width - 1,
                            math.floor(curr_pixel_update_x)
                        )
                        curr_pixel_update_y = min(
                            self.output_height - 1,
                            math.floor(curr_pixel_update_y)
                        )
                        heatmap[curr_pixel_update_y, curr_pixel_update_x, i] = max(
                            heatmap[curr_pixel_update_y, curr_pixel_update_x, i],
                            window[wy, wx]
                        )

        return heatmap.astype(float)


    def get_pafs(self, image_id):
        thickness = 1
        annotation_list = self.coco.get_annotation_list_by_image_id(image_id)
        bones = self.coco.get_bones(annotation_list)
        
        image_w, image_h = self.coco.get_width_height_by_image_id(image_id)

        heatmap_x = np.zeros((self.output_height, self.output_width, self.bones_amount))
        heatmap_y = np.zeros((self.output_height, self.output_width, self.bones_amount))

        for i in range(len(self.coco.filtered_bones_list)):
            bone_type = self.coco.filtered_bones_list[i][0]

            curr_heatmap_x = np.zeros((self.output_height, self.output_width))
            curr_heatmap_y = np.zeros((self.output_height, self.output_width))
            buffer_curr_heatmap = np.zeros((self.output_height, self.output_width))

            for [x1, y1], [x2, y2] in bones[bone_type]:
                x1 *= self.output_width / image_w
                y1 *= self.output_height / image_h
                x2 *= self.output_width / image_w
                y2 *= self.output_height / image_h
                x1 = min(self.output_width - 1, round(x1))
                y1 = min(self.output_height - 1, round(y1))
                x2 = min(self.output_width - 1, round(x2))
                y2 = min(self.output_height - 1, round(y2))

                x_v, y_v = (x2 - x1, y2 - y1)
                norm_v = ((x_v ** 2) + (y_v ** 2)) ** (1 / 2)
                x_uv, y_uv = x_v, y_v
                if norm_v != 0:
                    x_uv, y_uv = (x_uv / norm_v, y_uv / norm_v)

                if x_uv != 0:
                    curr_bone_heatmap_x = np.zeros((self.output_height, self.output_width))
                else:
                    curr_bone_heatmap_x = np.ones((self.output_height, self.output_width))
                if y_uv != 0:
                    curr_bone_heatmap_y = np.zeros((self.output_height, self.output_width))
                else:
                    curr_bone_heatmap_y = np.ones((self.output_height, self.output_width))
                curr_bone_heatmap_x = cv2.line(
                    curr_bone_heatmap_x, (x1, y1), (x2, y2), x_uv, thickness
                )
                curr_bone_heatmap_y = cv2.line(
                    curr_bone_heatmap_y, (x1, y1), (x2, y2), y_uv, thickness
                )

                i_curr_bone_heatmap_x = np.where(curr_bone_heatmap_x == x_uv)
                for i_y, i_x in zip(
                    i_curr_bone_heatmap_x[0], i_curr_bone_heatmap_x[1]
                ):
                    curr_heatmap_x[i_y, i_x] += x_uv
                    curr_heatmap_y[i_y, i_x] += y_uv
                    buffer_curr_heatmap[i_y, i_x] += 1

            buffer_curr_heatmap = np.array([
                [max(1, v) for v in row] for row in buffer_curr_heatmap
            ])
            curr_heatmap_x = curr_heatmap_x / buffer_curr_heatmap
            curr_heatmap_y = curr_heatmap_y / buffer_curr_heatmap

            heatmap_x[:, :, i] = curr_heatmap_x
            heatmap_y[:, :, i] = curr_heatmap_y
        
        return heatmap_x.astype(float), heatmap_y.astype(float)


    def get_images_id_by_amount_of_people(self, images_id, min_amount, max_amount):
        images_id_dict = {
            value: index for (index, value) in list(enumerate(images_id))
        }
        annotations_amount_per_id = np.zeros(len(images_id), dtype=int)
        for d in self.coco.data['annotations']:
            if d['image_id'] in images_id_dict:
                annotations_amount_per_id[images_id_dict[d['image_id']]] += 1

        selected_ids = [
            images_id_dict[images_id[aa_id]]
            for (aa_id, aa) in list(enumerate(annotations_amount_per_id))
            if aa >= min_amount and aa <= max_amount
        ]

        return selected_ids


    def build_generator(self):
        def _upsampling_block(x, index, bn_axis, filters):
            x = UpSampling2D(
                    size=(2, 2), name='upsampling_%s_upsampling' % index)(x)
            x = Conv2D(
                    filters=filters, kernel_size=(3, 3), strides=(1, 1),
                    padding='same', name='upsampling_%s_conv'
                        % index)(x)
            x = BatchNormalization(
                    axis=bn_axis, name='upsampling_%s_bn'
                        % index)(x)
            x = Activation('relu', name='upsampling_%s_relu' % index)(x)
            
            return x

        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1

        base_net = vgg19.VGG19(
            include_top=False, input_shape=self.input_shape, weights=None
        )
        base_net_output = base_net.get_layer('block5_pool').output
        base_net_output_filters = base_net_output.shape[bn_axis].value
        x = base_net_output

        for i in range(3):
            x = _upsampling_block(
                    x, i+1, bn_axis,
                    int(base_net_output_filters / (2 ** (i+1)))
                )

        keypoints = Conv2D(filters=self.keypoints_amount, kernel_size=(3, 3),
                strides=(1, 1), padding='same', activation='sigmoid',
                name='keypoints')(x)
        pafs = Conv2D(filters=2 * self.bones_amount, kernel_size=(3, 3),
                strides=(1, 1), padding='same', activation='tanh',
                name='paf')(x)

        x = Concatenate(name='slice')([keypoints, paf])

        return Model(inputs=base_net.input, outputs=[base_net.input, x])


    def build_discriminator(self):
        image_input = Input(shape=self.input_shape)
        heatmaps_input = Input(shape=self.output_shape)

        x_left_branch = Flatten(input_shape=self.input_shape)(image_input)
        x_right_branch = Flatten(input_shape=self.input_shape)(heatmaps_input)

        x = Concatenate()([x_left_branch, x_right_branch])
        x = Dense(64)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(1, activation='sigmoid')(x)

        return Model(inputs=[image_input, heatmaps_input], outputs=x)


    def train(self, epochs, batch_size):
        '''
        get all images' filename
        '''
        images_filename = [
            image_filename
            for image_filename in os.listdir(self.images_dir)
            if os.path.isfile(os.path.join(self.images_dir, image_filename))
        ]
        
        '''
        get all images' COCO id
        '''
        images_id = [
            self.coco.get_image_id_by_image_filename(image_filename)
            for image_filename in images_filename
        ]

        '''
        select only images that contains at least one person
        '''
        images_id_by_amount_of_people = self.get_images_id_by_amount_of_people(
            images_id, 1, 10000
        )
        images_filename = list(
            images_filename[i] for i in images_id_by_amount_of_people
        )
        images_id = list(images_id[i] for i in images_id_by_amount_of_people)
        number_of_images = min(
            len(images_id_by_amount_of_people),
            int(total_amount_of_images)
        )

        images_filename = images_filename[:number_of_images]
        images_id = images_id[:number_of_images]

        '''
        # Adversarial ground truths
        '''
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            '''
            Train Discriminator
            '''

            '''
            Select a random batch of images
            '''
            idx = np.random.randint(0, len(images_id), batch_size)
            imgs_ids = images_id[idx]
            real_imgs = [
                [
                    self.get_input_image(
                        os.path.join(
                            self.images_dir,
                            images_filename[random_index]
                        )
                    )
                    ,
                    np.concatenate(
                        (
                            self.get_heatmaps(images_id[random_index]),
                            self.get_pafs(images_id[random_index])
                        ),
                        axis=2)
                ]
                for random_index in imgs_ids
            ]

            '''
            Generate a batch of new images
            '''
            fake_imgs = self.generator.predict(real_imgs[:,0])

            '''
            Train the discriminator
            '''
            d_loss_real = self.discriminator.train_on_batch(real_imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(fake_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            '''
            Train Generator
            '''

            '''
            Train the generator (to have the discriminator label samples as valid)
            '''
            g_loss = self.combined.train_on_batch(real_imgs[:,0], valid)

            '''
            Plot the progress
            '''
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=30000, batch_size=1)
