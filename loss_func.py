import tensorflow as tf
import config as cfg
from utils import cal_iou


def yolo_loss(y_true, y_pred):
    """

    :param y_true: [batch_size, 7, 7, 25]
    :param y_pred: [batch_size, 7, 7, 30]
    :return:
    """
    # 类别标签
    _classes = y_pred[..., 10:]
    classes = y_true[..., 5:]
    # (batch_size, 7, 7, 2)
    _confidences = y_pred[..., 8:10]
    # (batch_size, 7, 7, 1)
    confidences = y_true[..., 4:5]

    # (batch_size, 7, 7, 4)
    bboxes = y_true[..., 0:4]
    # (batch_size, 7, 7, 1, 4)
    bboxes = tf.reshape(bboxes, (-1, cfg.CELL_SIZE, cfg.CELL_SIZE, 1, 4))
    _bboxes = y_pred[..., 0:8]
    # (batch_size, 7, 7, 2, 4)
    _bboxes = tf.reshape(_bboxes, (-1, cfg.CELL_SIZE, cfg.CELL_SIZE, cfg.B, 4))

    grid_x = tf.range(cfg.CELL_SIZE, dtype=tf.float32)
    grid_y = tf.range(cfg.CELL_SIZE, dtype=tf.float32)
    grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
    x_offset = tf.reshape(grid_x, (-1, 1))
    y_offset = tf.reshape(grid_y, (-1, 1))
    x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
    x_y_offset = tf.cast(tf.reshape(x_y_offset, [cfg.CELL_SIZE, cfg.CELL_SIZE, 1, 2]), tf.float32)

    # 将_bboxes转到原图上
    _bboxes_normal = tf.stack([
        (_bboxes[..., 0] + x_y_offset[..., 0]) / cfg.CELL_SIZE,
        (_bboxes[..., 1] + x_y_offset[..., 1]) / cfg.CELL_SIZE,
        tf.square(_bboxes[..., 2]),
        tf.square(_bboxes[..., 3]),
    ], axis=-1)

    # bboxes_ious: (n, 7, 7, 2)
    bboxes_ious = cal_iou(_bboxes_normal, bboxes)
    object_mask = tf.reduce_max(bboxes_ious, axis=-1, keep_dims=True)
    # 第i个cell第j个bbox负责产生损失
    object_mask = tf.cast(bboxes_ious >= object_mask, dtype=tf.float32) * confidences
    noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask

    # _bboxes[..., 0:2] = (_bboxes[..., 0:2] + x_y_offset) / cfg.CELL_SIZE
    # bboxes = bboxes[..., 0:2] * cfg.CELL_SIZE - x_y_offset
    # bboxes = tf.sqrt(bboxes[..., 2:4])
    bboxes_normal = tf.stack([
        bboxes[..., 0] * cfg.CELL_SIZE - x_y_offset[..., 0],
        bboxes[..., 1] * cfg.CELL_SIZE - x_y_offset[..., 1],
        tf.sqrt(bboxes[..., 2]),
        tf.sqrt(bboxes[..., 3]),
    ], axis=-1)

    object_delta = object_mask * (_confidences - bboxes_ious)
    object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3])) * cfg.OBJECT_SCALE

    onobject_delta = noobject_mask * _confidences
    nobject_loss = tf.reduce_mean(tf.reduce_sum(tf.square(onobject_delta), axis=[1, 2, 3])) * cfg.NOOBJECT_SCALE

    # 类别损失
    cls_delta = confidences * (classes - _classes)
    cls_loss = tf.reduce_mean(tf.reduce_sum(tf.square(cls_delta), axis=[1, 2, 3])) * cfg.CLASS_SCALE

    # 边框损失
    bbox_mask = tf.expand_dims(object_mask, axis=-1)
    bboxes_xy_delta = bbox_mask * (_bboxes[..., 0:2] - bboxes_normal[..., 0:2])
    bboxes_wh_delta = bbox_mask * (_bboxes[..., 2:4] - bboxes_normal[..., 2:4])
    bboxes_loss = tf.reduce_mean(tf.reduce_sum(tf.square(bboxes_xy_delta), axis=[1, 2, 3, 4])) * cfg.BBOX_SCALE + \
                  tf.reduce_mean(tf.reduce_sum(tf.square(bboxes_wh_delta), axis=[1, 2, 3, 4])) * cfg.BBOX_SCALE
    total_loss = cls_loss + object_loss + nobject_loss + bboxes_loss
    return total_loss

# y_true = tf.ones((32, 7, 7, 25))
# y_pred = tf.ones((32, 7, 7, 30))
#
# loss = yolo_loss(y_true, y_pred)
# print(loss)
