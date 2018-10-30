import numpy as np
import os
import json
import cv2
import re
import math


class CocoDataset:
    
    filtered_keypoints_list = np.array([
        'nose',
        'neck',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle',
    ])

    filtered_bones_list = np.array([
        ('nn',  ['nose', 'neck']),
        ('nls', ['neck', 'left_shoulder']),
        ('nrs', ['neck', 'right_shoulder']),
        ('lse', ['left_shoulder', 'left_elbow']),
        ('rse', ['right_shoulder', 'right_elbow']),
        ('lew', ['left_elbow', 'left_wrist']),
        ('rew', ['right_elbow', 'right_wrist']),
        ('nlh', ['neck', 'left_hip']),
        ('nrh', ['neck', 'right_hip']),
        ('lhk', ['left_hip', 'left_knee']),
        ('rhk', ['right_hip', 'right_knee']),
        ('lka', ['left_knee', 'left_ankle']),
        ('rka', ['right_knee', 'right_ankle']),
    ], dtype=object)
    
    '''
    filtered_keypoints_list = np.array([
        'nose',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle',
    ])
    
    filtered_bones_list = np.array([
        ('nls', ['nose', 'left_shoulder']),
        ('nrs', ['nose', 'right_shoulder']),
        ('lse', ['left_shoulder', 'left_elbow']),
        ('rse', ['right_shoulder', 'right_elbow']),
        ('lew', ['left_elbow', 'left_wrist']),
        ('rew', ['right_elbow', 'right_wrist']),
        ('nlh', ['nose', 'left_hip']),
        ('nrh', ['nose', 'right_hip']),
        ('lhk', ['left_hip', 'left_knee']),
        ('rhk', ['right_hip', 'right_knee']),
        ('lka', ['left_knee', 'left_ankle']),
        ('rka', ['right_knee', 'right_ankle']),
    ], dtype=object)
    '''
    
    gaussian_variance_list = {
        'nose': 0.026,
        'neck': 0.026,
        'left_shoulder': 0.079,
        'right_shoulder': 0.079,
        'left_elbow': 0.072,
        'right_elbow': 0.072,
        'left_wrist': 0.062,
        'right_wrist': 0.062,
        'left_hip': 0.107,
        'right_hip': 0.107,
        'left_knee': 0.087,
        'right_knee': 0.087,
        'left_ankle': 0.089,
        'right_ankle': 0.089
    }

    def __init__(self, mode, year, dataset_dir):
        self.mode = mode  # 'train' = train, 'val' = validation
        self.year = year
        self.annotation_type = 'keypoints'
        self.dataset_dir = dataset_dir

        self.annotations_path = os.path.join(
            self.dataset_dir, 'annotations/person_%s_%s%s.json' %
            (self.annotation_type, self.mode, self.year)
        )
        self.images_dir = os.path.join(
            self.dataset_dir, '%s%s' % (self.mode, self.year)
        )
        self.image_masks_dir = os.path.join(
            self.dataset_dir, '%smask%s' % (self.mode, self.year)
        )
        self.masked_images_dir = os.path.join(
            self.dataset_dir, '%smaskedimages%s' % (self.mode, self.year)
        )

        self.data = json.load(open(self.annotations_path))
        self.annotations = self.data['annotations']
        self.images = self.data['images']

        self.keypoints_list = np.array(self.data['categories'][0]['keypoints'])

    def generate_masks(self, verbose=True):
        from pycocotools.coco import COCO
        if not os.path.exists(self.image_masks_dir):
            os.makedirs(self.image_masks_dir)

        coco = COCO(self.annotations_path)
        ids = list(coco.imgs.keys())
        for i, img_id in enumerate(ids):
            ann_ids = coco.getAnnIds(imgIds=img_id)
            img_anns = coco.loadAnns(ann_ids)

            img_path = os.path.join(self.images_dir, '%012d.jpg' % img_id)
            mask_miss_path = os.path.join(
                self.image_masks_dir, 'mask_miss_%012d.png' % img_id
            )
            mask_all_path = os.path.join(
                self.image_masks_dir, 'mask_all_%012d.png' % img_id
            )

            img = cv2.imread(img_path)
            h, w, c = img.shape
            mask_all = np.zeros((h, w), dtype=np.uint8)
            mask_miss = np.zeros((h, w), dtype=np.uint8)

            flag = 0
            for p in img_anns:
                # seg = p['segmentation']

                if p['iscrowd'] == 1:
                    mask_crowd = coco.annToMask(p)
                    temp = np.bitwise_and(mask_all, mask_crowd)
                    mask_crowd = mask_crowd - temp
                    flag += 1
                    continue
                else:
                    mask = coco.annToMask(p)

                mask_all = np.bitwise_or(mask, mask_all)

                if p['num_keypoints'] <= 0:
                    mask_miss = np.bitwise_or(mask, mask_miss)

            if flag < 1:
                mask_miss = np.logical_not(mask_miss)
            elif flag == 1:
                mask_miss = np.logical_not(
                    np.bitwise_or(mask_miss, mask_crowd)
                )
                mask_all = np.bitwise_or(mask_all, mask_crowd)
            else:
                raise Exception('crowd segments > 1')

            cv2.imwrite(mask_miss_path, mask_miss * 255)
            cv2.imwrite(mask_all_path, mask_all * 255)

            if (i % 100 == 0):
                print('Processed %d of %d' % (i, len(ids)))

    def apply_masks(self, verbose=True):
        if not os.path.exists(self.masked_images_dir):
            os.makedirs(self.masked_images_dir)

        files = [
            (i, m)
            for (i, m) in zip(
                os.listdir(self.images_dir), os.listdir(self.image_masks_dir)
            )
        ]
        count = 1
        for (i, m) in files:
            img = cv2.imread(os.path.join(self.images_dir, i))
            mask = cv2.imread(os.path.join(self.image_masks_dir, m), 0)
            masked_image = cv2.bitwise_and(img, img, mask=mask)

            cv2.imwrite(
                # os.path.join(self.masked_images_dir, 'mask_all_%s' % i),
                os.path.join(self.masked_images_dir, i),
                masked_image
            )

            if verbose:
                if count % 100 == 0:
                    print('Processed %d of %d' % (count, len(files)))

            count += 1

    def get_image_id_by_image_filename(self, image_filename):
        image_id = int(re.sub('\\D', '', image_filename))

        return image_id

    def get_annotation_list_by_image_id(self, image_id):
        annotation_list = [
            d for d in self.annotations if d['image_id'] == image_id
        ]

        return annotation_list
    
    def get_mask_per_person_by_image_id(self, image_id):
        annotation_list = self.get_annotation_list_by_image_id(image_id)
        segmentations = self.get_segmentations(annotation_list)
        image_w, image_h = self.get_width_height_by_image_id(image_id)
    
        mask_per_person = np.zeros((image_h, image_w))
    
        for person_index in range(len(segmentations)):
            for segmentation in segmentations[person_index]:
                reshaped_segmentation = [
                    np.array(
                        segmentation, np.float64
                    ).reshape((-1, 1, 2)).astype(int)
                ]
                mask_per_person = cv2.fillPoly(
                    mask_per_person, reshaped_segmentation, person_index + 1
                )
    
        return mask_per_person
    
    def get_mask_not_per_person_by_image_id(self, image_id):
        mask_per_person = self.get_mask_per_person_by_image_id(image_id)
        mask_not_per_person = np.vectorize(
                lambda p: min(math.ceil(p), 1)
        )(mask_per_person)
        
        return mask_not_per_person
    
    def get_maskfile_by_image_id(self, image_id):
        image_id_str = str(image_id)
        image_filepath = os.path.join(
            self.image_masks_dir,
            'mask_all_%s.png' % image_id_str.zfill(12)
        )
        mask = cv2.imread(image_filepath, 0)
        mask = mask.astype(float)
        
        return mask

    def get_areas(self, annotation_list):
        areas = [annotation['area'] for annotation in annotation_list]

        return areas

    def get_width_height_by_image_id(self, image_id):
        image = [image for image in self.images if image['id'] == image_id][0]

        return image['width'], image['height']

    def get_keypoints(self, annotation_list):
        '''
        getting keypoints from .json file.
        'keypoints' variable is a dictionary, where each key is a keypoint,
        and each dict value is a list of [x, y] positions of that keypoint.
        '''
        keypoints = {
            self.keypoints_list[i]: [
                annotation['keypoints'][3 * i:3 * i + 2]
                for annotation in annotation_list
            ] for i in range(len(self.keypoints_list))
        }
        '''
        keypoints' values are filtered; if its value is [0, 0], it is
        considered as None.
        '''
        keypoints = {
            keypoint_name: [
                keypoint_value if keypoint_value != [0, 0] else None
                for keypoint_value in keypoint_values
            ] for (keypoint_name, keypoint_values) in keypoints.items()
        }
        '''
        keypoints' keys are filtered; if its key is not in
        self.filtered_keypoints_list, it is removed.
        '''
        keypoints = {
            keypoint_name: keypoint_values
            for (keypoint_name, keypoint_values) in keypoints.items()
            if keypoint_name in self.filtered_keypoints_list
        }
        '''
        add 'neck' keypoint, which does not exist in COCO-Dataset.
        it is calculated as the average of left and right shoulders.
        '''
        keypoints['neck'] = [
            [
                round((ls_rs[0][0] + ls_rs[1][0]) / 2),
                round((ls_rs[0][1] + ls_rs[1][1]) / 2)
            ]
            if (ls_rs[0] is not None and ls_rs[1] is not None) else None
            for (ls_rs)
            in zip(keypoints['left_shoulder'], keypoints['right_shoulder'])
        ]

        return keypoints

    def get_keypoints_with_v(self, annotation_list):
        keypoints = {
            self.keypoints_list[i]: [
                annotation['keypoints'][3 * i:3 * i + 3]
                for annotation in annotation_list
            ] for i in range(len(self.keypoints_list))
        }
        keypoints = {
            keypoint_name: keypoint_values
            for (keypoint_name, keypoint_values) in keypoints.items()
            if keypoint_name in self.filtered_keypoints_list
        }
        keypoints['neck'] = [
            [
                round((ls_rs[0][0] + ls_rs[1][0]) / 2),
                round((ls_rs[0][1] + ls_rs[1][1]) / 2)
            ]
            if (ls_rs[0] is not None and ls_rs[1] is not None) else None
            for (ls_rs)
            in zip(keypoints['left_shoulder'], keypoints['right_shoulder'])
        ]

        return keypoints

    def get_bones(self, annotation_list):
        keypoints = self.get_keypoints(annotation_list)
        bones = {
            bone_name: [
                [k1, k2]
                for (k1, k2) in zip(
                    keypoints[
                        self.filtered_bones_list[
                            self.filtered_bones_list[:, 0] == bone_name
                        ][0][1][0]],
                    keypoints[
                        self.filtered_bones_list[
                            self.filtered_bones_list[:, 0] == bone_name
                        ][0][1][1]]
                ) if k1 is not None and k2 is not None
            ] for bone_name in self.filtered_bones_list[:, 0]
        }

        return bones

    def get_segmentations(self, annotation_list):
        '''
        segmentations = [
            [
                [segmentation[i:i + 2] for i in range(0, len(segmentation), 2)]
                for segmentation in annotation['segmentation']
            ] for annotation in annotation_list
        ]
        '''
        segmentations = []
        for annotation in annotation_list:
            curr_annotation = []
            for ann_segmentation in annotation['segmentation']:
                curr_segmentation = []
                for i in range(0, len(ann_segmentation), 2):
                    x, y = ann_segmentation[i: i + 2]
                    if isinstance(x, float) and isinstance(y, float):
                        curr_segmentation += [[x, y]]
                curr_annotation += [curr_segmentation]
            segmentations += [curr_annotation]
        
        return segmentations
