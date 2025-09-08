import numpy as np
import time
import sys
import cv2
import os
import matplotlib.pyplot as plt
from datetime import datetime
from Layers import LayerList, Convolution, LinearLayer, Flatten
from Activations import ReLu, Softmax, Tanh, Sigmoid, MaxPooling
from Loss_and_Optimizer import MeanSquaredError, CrossEntropyLoss, SGD, SGDWithMomentum
from SSD import *
from Prepare_Data import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
os.environ["OMP_NUM_THREADS"] = "8"

VGG_16 = LayerList(Convolution(num_kernels=64, kernel_size=3, padding=1, stride=1, input_depth=3),
                   Convolution(num_kernels=64, kernel_size=3, padding=1, stride=1, input_depth=64),
                   MaxPooling(),

                   Convolution(num_kernels=128, kernel_size=3, padding=1, stride=1, input_depth=64),
                   Convolution(num_kernels=128, kernel_size=3, padding=1, stride=1, input_depth=128),
                   MaxPooling(),

                   Convolution(num_kernels=256, kernel_size=3, padding=1, stride=1, input_depth=128),
                   Convolution(num_kernels=256, kernel_size=3, padding=1, stride=1, input_depth=256),
                   Convolution(num_kernels=256, kernel_size=3, padding=1, stride=1, input_depth=256),
                   MaxPooling(),

                   Convolution(num_kernels=512, kernel_size=3, padding=1, stride=1, input_depth=256),
                   Convolution(num_kernels=512, kernel_size=3, padding=1, stride=1, input_depth=512),
                   Convolution(num_kernels=512, kernel_size=3, padding=1, stride=1, input_depth=512),
                   MaxPooling(),

                   Convolution(num_kernels=512, kernel_size=3, padding=1, stride=1, input_depth=512),
                   Convolution(num_kernels=512, kernel_size=3, padding=1, stride=1, input_depth=512),
                   Convolution(num_kernels=512, kernel_size=3, padding=1, stride=1, input_depth=512),
                   MaxPooling(),

                   Flatten(),

                   LinearLayer(7 * 7 * 512, 4096, activation_func=ReLu),
                   LinearLayer(4096, 4096, activation_func=ReLu),
                   LinearLayer(4096, 4, activation_func=Softmax)
                   )

Grey_Test_VGG_16 = LayerList(
    Convolution(num_kernels=8, kernel_size=3, padding=1, stride=1, input_depth=1),
    Convolution(num_kernels=8, kernel_size=3, padding=1, stride=1, input_depth=8),
    MaxPooling(kernel_size=2, stride=2),

    Convolution(num_kernels=16, kernel_size=3, padding=1, stride=1, input_depth=8),
    Convolution(num_kernels=16, kernel_size=3, padding=1, stride=1, input_depth=16),
    MaxPooling(kernel_size=2, stride=2),

    Flatten(),

    LinearLayer(7 * 7 * 16, 128, activation_func=ReLu),
    LinearLayer(128, 10, activation_func=Softmax)
)

Linear_Model = LayerList(
    Flatten(),

    LinearLayer(28 * 28, 512, activation_func=ReLu),
    LinearLayer(512, 256, activation_func=ReLu),
    LinearLayer(256, 128, activation_func=ReLu),
    LinearLayer(128, 10, activation_func=Softmax)
)

SSD_Test_VGG_16 = LayerList(
    Convolution(num_kernels=16, kernel_size=3, padding=1, stride=1, input_depth=3),
    MaxPooling(kernel_size=2, stride=2),

    Convolution(num_kernels=32, kernel_size=3, padding=1, stride=1, input_depth=16),
    MaxPooling(kernel_size=2, stride=2),

    Convolution(num_kernels=64, kernel_size=3, padding=1, stride=1, input_depth=32),
    MaxPooling(kernel_size=2, stride=2),
    SSDHead(in_channels=64, num_priors=16, num_classes=2),

    Convolution(num_kernels=128, kernel_size=3, padding=1, stride=1, input_depth=64),
    MaxPooling(kernel_size=2, stride=2),
    SSDHead(in_channels=128, num_priors=16, num_classes=2),

    Convolution(num_kernels=256, kernel_size=3, padding=1, stride=1, input_depth=128),
    MaxPooling(kernel_size=2, stride=2),
    SSDHead(in_channels=256, num_priors=16, num_classes=2),
)


if __name__ == "__main__":

    with open('hand_data.pkl', 'rb') as f:
        img_array, raw_bboxes = pickle.load(f)

    with open('anchor_boxes.npy', 'rb') as f:
        anchors = np.load(f)


    CNN_Model = SSD_Test_VGG_16
    optimizer = SGDWithMomentum(alpha=.01, beta=0.9, decay=.0005, decay_type="exponential")
    loss_fn = CrossEntropyLoss()
    box_loss_fn = SmoothL1()

    num_epochs = 1
    batch_size = 32

    img_input = [img.transpose(2, 0, 1) / 255.0 for img in img_array]

    cls_targets = []
    loc_targets = []

    for img_bboxes in raw_bboxes:  # raw_bboxes is a list of boxes per image
        gt_boxes = []
        gt_labels = []

        for x_min, y_min, x_max, y_max, _cls in img_bboxes:
            bb = BoundingBox(None, x_min, y_min, x_max, y_max)
            bb.normalize(1920, 1080)
            gt_boxes.append(bb.bbox)
            gt_labels.append(1)  # all hand boxes labeled as class 1

        gt_boxes = np.array(gt_boxes)
        gt_labels = np.array(gt_labels)

        # Match anchors to ground truth
        labels, matched = match_anchors_to_gt(anchors, gt_boxes, gt_labels)

        # Encode offsets
        encoded = encode_offsets(anchors, matched)

        cls_targets.append(labels)
        loc_targets.append(encoded)

    cls_targets = np.array(cls_targets)
    loc_targets = np.array(loc_targets)

    start_time = time.time()
    for epoch in range(num_epochs):
        total_loss = 0

        for start in range(0, len(img_input), batch_size):
            end = start + batch_size
            batch_imgs = img_input[start:end]
            batch_loc_targets = loc_targets[start:end]
            batch_cls_targets = cls_targets[start:end]

            loc_preds, cls_preds, feature_maps = CNN_Model(batch_imgs)

            pos_mask = batch_cls_targets > 0  # shape (B, A)
            valid_mask = batch_cls_targets >= 0  # shape (B, A)

            box_loss = box_loss_fn(loc_preds[pos_mask], batch_loc_targets[pos_mask])

            n_classes = cls_preds.shape[2]
            one_hot = np.eye(n_classes)[batch_cls_targets]

            class_loss = loss_fn(cls_preds[valid_mask], one_hot[valid_mask])

            total_loss += (box_loss + class_loss)


            box_grad = np.zeros_like(loc_preds)
            class_grad = np.zeros_like(cls_preds)

            box_grad[pos_mask] = box_loss_fn.derivative(loc_preds[pos_mask], batch_loc_targets[pos_mask])
            class_grad[valid_mask] = loss_fn.derivative(cls_preds[valid_mask], one_hot[valid_mask])

            # Backward pass and update
            CNN_Model.backward((class_grad, box_grad))
            CNN_Model.update(optimizer, epoch)
        end_time = time.time()
        time = end_time - start_time

        print(f"Epoch {epoch} completed in {time/60: .2f} minuets: Loss = {total_loss:.4f}")


def draw_bounding_boxes(image, bboxes, anchors):

    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax
        if xmin < 0 or ymin < 0 or xmax > IMG_SIZE or ymax > IMG_SIZE:
            print(f"Warning: Bounding box out of bounds: {xmin}, {ymin}, {xmax}, {ymax}")
        image = cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
    '''
    anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax = anchors[400]
    image = cv2.rectangle(image, (int(anchor_xmin), int(anchor_ymin)),
                          (int(anchor_xmax), int(anchor_ymax)),
                          (0, 0, 255), 1)
    '''
    for anchor in anchors:
        anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax = anchor
        image = cv2.rectangle(image, (int(anchor_xmin), int(anchor_ymin)),
                                      (int(anchor_xmax), int(anchor_ymax)),
                                      (0, 0, 255), 1)
    return image


def visualize_image_with_boxes_from_array(img_array, bb_objects, anchors):

    img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img_with_boxes = draw_bounding_boxes(img_resized, bb_objects, anchors)
    cv2.imshow("Image with Bounding Boxes", img_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''
for img, bboxes in zip(img_array, raw_bboxes):
    bb_objects = []
    for x_min, y_min, x_max, y_max, _cls in bboxes:
        bb = BoundingBox(None,x_min, y_min, x_max, y_max)
        bb.normalize(1920, 1080)
        bb_objects.append(bb)

    visualize_image_with_boxes_from_array(img, bb_objects, anchors)
'''
