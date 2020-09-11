import tensorflow as tf
import numpy as np
from tensorflow.keras import (
models, layers, datasets, Model, regularizers
)
yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        dtype=np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])


def darknet_conv(x_in, filter_num, filter_size, strides=1, batch_norm=True):
    if strides == 1:
        padding = 'same'
    else:
        x_in = layers.ZeroPadding2D(((1, 0), (1, 0)))(x_in)   # 使kernel_size=3, strides=2时，特征图减小一半
        padding = 'valid'
    x_in = layers.Conv2D(filers=filter_num, kernel_size=filter_size, strides=strides,
                         padding=padding, use_bias=not batch_norm,
                         kernel_regularizer=regularizers.l2(0.0005))(x_in)
    if batch_norm:
        x_in = layers.BatchNormalization()(x_in)
        x_out = layers.LeakyReLU(alpha=0.1)(x_in)
    return x_out


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
            inputs = layers.Input(x_in[0].shape[1:], x_in[1].shape[1:])
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
        x = layers.Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
                                                   anchor_num, class_num + 5)))(x)
        return Model(inputs, x, name=name)(x_in)
    return yoloHead


# pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
def yolo_boxes(pred, anchors, class_num):
    pass


def yolo_nms(outputs):
    pass


def yolo_v3(size, channels=3, anchors=yolo_anchors,
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
    pass
