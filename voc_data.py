import numpy as np
from exception import ValueValidException
from data_augment import DataAugment
import config as cfg
import cv2

# load config
cell_size = cfg.CELL_SIZE
im_size = cfg.IM_SIZE
w, h = cfg.IM_SIZE
img_channel_mean = cfg.IMG_CHANNEL_MEAN
img_scaling_factor = cfg.IMG_SCALING_FACTOR
classes_num = cfg.CLASSES_NUM


class VocData(object):
    def _parse_annotation(self, annotation):
        lines = annotation.strip().split()
        image_path = lines[0]
        gt_boxes = np.asarray([list(map(float, box.split(','))) for box in lines[1:]])
        return image_path, gt_boxes

    def data_generator_wrapper(self, annotations, batch_size):
        n = len(annotations)
        if n == 0 or batch_size <= 0:
            raise ValueValidException('样本数量为0或者batch_size小于等于0, please check it')
        return self._data_generator(annotations, batch_size)

    def _get_label(self, gt_boxes):
        label = np.zeros(shape=(cell_size, cell_size, (4 + 1) + classes_num), dtype=np.float32)
        i = np.floor(gt_boxes[:, 0] * cell_size).astype(np.int32)
        j = np.floor(gt_boxes[:, 1] * cell_size).astype(np.int32)
        label[i, j, 0:4] = gt_boxes[:, 0:4]
        # 有物体置信度1  没有为0
        label[i, j, 4] = 1
        class_index = gt_boxes[:, 4].astype(np.int32)
        label[i, j, 4 + class_index] = 1
        return label

    def _data_generator(self, annotations, batch_size):
        i = 0
        n = len(annotations)
        data_augment = DataAugment(augment=True, horizontal_flip=True, vertical_flip=True)
        while True:
            img_data = []
            labels = []
            for b in range(batch_size):
                if i == 0:
                    x = np.random.permutation(n)
                    annotations = annotations[x]
                annotation = annotations[i]
                image_path, gt_boxes = self._parse_annotation(annotation)
                # TODO 数据增广
                # img, gt_boxes = data_augment(image_path, gt_boxes)
                # BGR -> RGB
                img = cv2.imread(image_path)
                img = img[:, :, (2, 1, 0)]
                img = img.astype(np.float32)
                img[:, :, 0] -= img_channel_mean[0]
                img[:, :, 1] -= img_channel_mean[1]
                img[:, :, 2] -= img_channel_mean[2]
                img /= img_scaling_factor
                # 原图宽高
                height, width = img.shape[:2]
                # gt_boxes
                #  进行归一化到0-1
                ctr_xy = (gt_boxes[:, [0, 1]] + gt_boxes[:, [2, 3]]) // 2
                bbox_wh = gt_boxes[:, [2, 3]] - gt_boxes[:, [0, 1]]
                gt_boxes[:, [0, 1]] = np.minimum(ctr_xy / [width, height], 1)
                gt_boxes[:, [2, 3]] = np.minimum((bbox_wh / [width, height]), 1)
                # print(gt_boxes)
                label = self._get_label(gt_boxes)
                # resize
                img = cv2.resize(img, im_size, interpolation=cv2.INTER_CUBIC)
                img_data.append(img)
                labels.append(label)
                i = (i + 1) % n
            img_data = np.asarray(img_data)
            labels = np.asarray(labels)
            yield img_data, labels
