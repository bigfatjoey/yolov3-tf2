import tensorflow as tf
import numpy as np
from tensorflow.keras import (
    layers, Model, regularizers
)
from tensorflow.keras.losses import (
    binary_crossentropy, sparse_categorical_crossentropy
)
from utils import broadcast_iou
yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        dtype=np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
yolo_max_boxes = 100
yolo_iou_threshold = 0.5
yolo_score_threshold = 0.5


def darknet_conv(x_in, filter_num, filter_size, strides=1, batch_norm=True):
    if strides == 1:
        padding = 'same'
    else:
        x_in = layers.ZeroPadding2D(((1, 0), (1, 0)))(x_in)   # 使kernel_size=3, strides=2时，特征图减小一半
        padding = 'valid'
    x_in = layers.Conv2D(filters=filter_num, kernel_size=filter_size, strides=strides,
                         padding=padding, use_bias=not batch_norm,
                         kernel_regularizer=regularizers.l2(0.0005))(x_in)        # kernel_regularizer=regularizers.l2(0.0005)
    if batch_norm:
        x_in = layers.BatchNormalization()(x_in)
        x_in = layers.LeakyReLU(alpha=0.1)(x_in)
    return x_in


def darknet_residual(x_in, filter_num):  # darknet残差块
    pre = x_in
    x = darknet_conv(x_in, filter_num // 2, 1)
    x = darknet_conv(x, filter_num, 3)
    x = layers.Add()([pre, x])
    return x


def darknet_block(x_in, filter_num, blocks):  # darknet块
    x = darknet_conv(x_in, filter_num, 3, strides=2)
    for _ in range(blocks):
        x = darknet_residual(x, filter_num)
    return x


def darknet(name=None):
    x = inputs = layers.Input([None, None, 3])
    x = darknet_conv(x, 32, 3)
    x = darknet_block(x, 64, 1)
    x = darknet_block(x, 128, 2)
    x = x_1 = darknet_block(x, 256, 8)
    x = x_2 = darknet_block(x, 512, 8)
    x_3 = darknet_block(x, 1024, 4)
    return Model(inputs, (x_1, x_2, x_3), name=name)


def yolo_neck(filter_num, name=None):
    def yoloNeck(x_in):
        if isinstance(x_in, tuple):
            inputs = layers.Input(x_in[0].shape[1:]), layers.Input(x_in[1].shape[1:])
            x_down, x_up = inputs
            x_down = darknet_conv(x_down, filter_num, 1)
            x_down = layers.UpSampling2D(2)(x_down)
            x = layers.Concatenate()([x_down, x_up])
        else:
            x = inputs = layers.Input(x_in.shape[1:])

        x = darknet_conv(x, filter_num, 1)
        x = darknet_conv(x, filter_num*2, 3)
        x = darknet_conv(x, filter_num, 1)
        x = darknet_conv(x, filter_num*2, 3)
        x = darknet_conv(x, filter_num, 1)
        return Model(inputs, x, name=name)(x_in)
    return yoloNeck


# the output of yolo network
def yolo_head(filter_num, anchor_num, class_num, name=None):
    def yoloHead(x_in):
        x = inputs = layers.Input(x_in.shape[1:])
        x = darknet_conv(x, filter_num*2, 3)
        x = darknet_conv(x, anchor_num*(class_num + 5), 1, batch_norm=False)
        x = layers.Lambda(lambda xt: tf.reshape(xt, (-1, tf.shape(xt)[1], tf.shape(xt)[2],
                                                anchor_num, class_num + 5)))(x)
        return Model(inputs, x, name=name)(x_in)
    return yoloHead


# pred: (batch_size, grid, grid, anchor_num, (x, y, w, h, obj, classe_num))
def yolo_boxes(pred, anchors, class_num):
    grid_size = tf.shape(pred)[1:3]
    box_xy, box_wh, objectness, class_probs = tf.split(
        pred, (2, 2, 1, class_num), axis=-1
    )
    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # 偏移量，用于计算loss

    grid = tf.meshgrid(tf.range(grid_size[1]), tf.range(grid_size[0]))  # grid[x][y] == (y, x)
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)   # (grid, grid, 1, 2) 与box_xy维度匹配

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)  # 计算偏移量+格点位置
    box_wh = tf.exp(box_wh) * anchors  # 预测框的宽高

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box   # 返回预测框的形状bbox，置信度objectness，各类概率class_probs，预测偏移量pred_box


def yolo_nms(outputs):  # non maximum suppression
    b, c, t = [], [], []  # box，confidence，type

    # list b :bbox dimension (batch_size, grid*grid*anchor_num, 4)
    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)

    scores = confidence * class_probs  # 得分是confidence乘以分类
    # Greedily selects a subset of bounding boxes in descending order of score.
    # 'nmsed_boxes': A [batch_size, max_detections, 4] float32 tensor containing the non-max suppressed boxes.
    # 'nmsed_scores': A [batch_size, max_detections] float32 tensor containing the scores for the boxes.
    # 'nmsed_classes': A [batch_size, max_detections] float32 tensor containing the class for boxes.
    nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(
            scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=yolo_max_boxes,
        max_total_size=yolo_max_boxes,
        iou_threshold=yolo_iou_threshold,
        score_threshold=yolo_score_threshold
    )

    return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections


def yolo_v3(size=None, channels=3, anchors=yolo_anchors,
            masks=yolo_anchor_masks, class_num=80, training=False):
    x = inputs = layers.Input([size, size, channels], name='input')
    x_1, x_2, x_3 = darknet(name='darknet')(x)
    x = yolo_neck(512, name='yolo_neck_3')(x_3)
    output_3 = yolo_head(512, 3, class_num, name='output_3')(x)

    x = yolo_neck(256, name='yolo_neck_2')((x, x_2))
    output_2 = yolo_head(256, 3, class_num, name='output_2')(x)

    x = yolo_neck(128, name='yolo_neck_1')((x, x_1))
    output_1 = yolo_head(128, 3, class_num, name='output_1')(x)

    if training:
        return Model(inputs, (output_3, output_2, output_1), name='yolov3')

    boxes_3 = layers.Lambda(lambda xt: yolo_boxes(xt, anchors[masks[0]], class_num),
                            name='yolo_boxes_3')(output_3)
    boxes_2 = layers.Lambda(lambda xt: yolo_boxes(xt, anchors[masks[1]], class_num),
                            name='yolo_boxes_2')(output_2)
    boxes_1 = layers.Lambda(lambda xt: yolo_boxes(xt, anchors[masks[2]], class_num),
                            name='yolo_boxes_1')(output_1)

    outputs = layers.Lambda(lambda xm: yolo_nms(xm),
                            name='yolo_nms')((boxes_3[:3], boxes_2[:3], boxes_1[:3]))

    return Model(inputs, outputs, name='yolov3')


# 按照yolo_anchors[mask],各大小特征图的anchor box分别计算loss
def yolo_loss(anchors, class_num=80, ignore_thresh=0.5):
    def yoloLoss(y_true, y_pred):
        pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(
            y_pred, anchors, class_num)
        pred_xy = pred_xywh[..., 0:2]  # 取出偏移量
        pred_wh = pred_xywh[..., 2:4]

        true_box, true_obj, true_class_idx = tf.split(
            y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - \
                  tf.cast(grid, tf.float32)
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh),
                           tf.zeros_like(true_wh), true_wh)  # 将真实坐标转变为偏移量用于计算loss

        obj_mask = tf.squeeze(true_obj, -1)
        # ignore false positive when iou is over threshold
        best_iou = tf.map_fn(
            lambda x: tf.reduce_max(broadcast_iou(x[0], tf.boolean_mask(
                x[1], tf.cast(x[2], tf.bool))), axis=-1),
            (pred_box, true_box, obj_mask),
            tf.float32)
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        xy_loss = obj_mask * box_loss_scale * \
                  tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * \
                  tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        obj_loss = binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + \
                   (1 - obj_mask) * ignore_mask * obj_loss
        # TODO: use binary_crossentropy instead
        class_loss = obj_mask * sparse_categorical_crossentropy(
            true_class_idx, pred_class)

        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        return xy_loss + wh_loss + obj_loss + class_loss
    return yoloLoss


