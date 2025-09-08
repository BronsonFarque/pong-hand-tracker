import numpy as np
from Layers import LayerList, Convolution, SSDHead
from Loss_and_Optimizer import CrossEntropyLoss, SmoothL1, SGDWithMomentum
from Activations import *
anchors = np.load("anchor_boxes.npy")


VGG_16_SSD = LayerList(
    # Block 1
    Convolution(num_kernels=64, kernel_size=3, padding=1, stride=1, input_depth=3),
    Convolution(num_kernels=64, kernel_size=3, padding=1, stride=1, input_depth=64),
    MaxPooling(kernel_size=2, stride=2),  # 1/2 size

    # Block 2
    Convolution(num_kernels=128, kernel_size=3, padding=1, stride=1, input_depth=64),
    Convolution(num_kernels=128, kernel_size=3, padding=1, stride=1, input_depth=128),
    MaxPooling(kernel_size=2, stride=2),  # 1/4 size

    # Block 3
    Convolution(num_kernels=256, kernel_size=3, padding=1, stride=1, input_depth=128),
    Convolution(num_kernels=256, kernel_size=3, padding=1, stride=1, input_depth=256),
    Convolution(num_kernels=256, kernel_size=3, padding=1, stride=1, input_depth=256),
    MaxPooling(kernel_size=2, stride=2),  # 1/8 size

    # Block 4
    Convolution(num_kernels=512, kernel_size=3, padding=1, stride=1, input_depth=256),
    Convolution(num_kernels=512, kernel_size=3, padding=1, stride=1, input_depth=512),
    Convolution(num_kernels=512, kernel_size=3, padding=1, stride=1, input_depth=512),
    MaxPooling(kernel_size=2, stride=2),  # 1/16 size

    # Block 5
    Convolution(num_kernels=512, kernel_size=3, padding=1, stride=1, input_depth=512),
    Convolution(num_kernels=512, kernel_size=3, padding=1, stride=1, input_depth=512),
    Convolution(num_kernels=512, kernel_size=3, padding=1, stride=1, input_depth=512),
    MaxPooling(kernel_size=3, stride=1, padding=1),

    # Converted fc6 (fully connected → convolution)
    Convolution(num_kernels=1024, kernel_size=3, padding=6, stride=1, dilation=6, input_depth=512),
    SSDHead(1024, 9, 2),

    # Converted fc7
    Convolution(num_kernels=1024, kernel_size=1, padding=0, stride=1, input_depth=1024),
    SSDHead(1024, 9, 2),

    # Extra Feature Maps
    Convolution(num_kernels=512, kernel_size=3, padding=1, stride=2, input_depth=1024), # 19x19
    SSDHead(512, 9, 2),
    Convolution(num_kernels=256, kernel_size=3, padding=1, stride=2, input_depth=512),  # 10x10
    SSDHead(256, 9, 2),
    Convolution(num_kernels=256, kernel_size=3, padding=1, stride=2, input_depth=256),  # 5x5
    SSDHead(256, 9, 2),
    Convolution(num_kernels=256, kernel_size=3, padding=1, stride=1, input_depth=256),  # 3x3
    SSDHead(256, 9, 2),
    Convolution(num_kernels=256, kernel_size=3, padding=1, stride=1, input_depth=256),  # 1x1
    SSDHead(256, 9, 2)
)


def jaccard_overlap(anchors, gt_boxes):
    """
    anchors:  (N,4) array of [x1,y1,x2,y2]
    gt_boxes: (M,4) array of [x1,y1,x2,y2]
    returns:  (N,M) array of IoUs
    """

    A = anchors[:, None, :]   # (N,1,4)
    G = gt_boxes[None, :, :]  # (1,M,4)

    # intersection coords
    x1 = np.maximum(A[..., 0], G[..., 0])
    y1 = np.maximum(A[..., 1], G[..., 1])
    x2 = np.minimum(A[..., 2], G[..., 2])
    y2 = np.minimum(A[..., 3], G[..., 3])

    # intersection area
    inter_w = np.clip(x2 - x1, 0, None)
    inter_h = np.clip(y2 - y1, 0, None)
    inter   = inter_w * inter_h

    # areas
    area_a = (A[..., 2] - A[..., 0]) * (A[..., 3] - A[..., 1])  # (N,1)
    area_g = (G[..., 2] - G[..., 0]) * (G[..., 3] - G[..., 1])  # (1,M)

    # union
    union = area_a + area_g - inter

    return inter / np.where(union > 0, union, 1)  # (N,M)


def match_anchors_to_gt(anchors, gt_boxes, gt_labels,
                        pos_thresh=0.5, neg_thresh=0.4):
    """
    anchors:     (N,4)
    gt_boxes:    (M,4)
    gt_labels:   (M,)  labels for each ground‐truth box
    pos_thresh:  IoU ≥ pos_thresh → positive
    neg_thresh:  IoU <  neg_thresh → negative
    returns:     labels (N,) where
                   >0 = class,
                    0 = background,
                   -1 = ignore
    """
    N = anchors.shape[0]
    labels = np.full((N,), 1, dtype=int)

    ious = jaccard_overlap(anchors, gt_boxes)     # (N,M)

    best_anchor_for_gt = np.argmax(ious, axis=0)
    labels[best_anchor_for_gt] = gt_labels

    best_gt_for_anchor = np.argmax(ious, axis=1)  # (N,)
    best_iou_for_anchor = ious.max(axis=1)       # (N,)

    # positives
    pos_idxs = best_iou_for_anchor >= pos_thresh
    labels[pos_idxs] = gt_labels[best_gt_for_anchor[pos_idxs]]

    # negatives
    neg_idxs = best_iou_for_anchor < neg_thresh
    labels[neg_idxs] = 0
    matched_boxes = np.zeros((N, 4))
    matched_boxes[best_anchor_for_gt] = gt_boxes
    matched_boxes[pos_idxs] = gt_boxes[best_gt_for_anchor[pos_idxs]]
    return labels, matched_boxes


def encode_offsets(anchors, matched_boxes):

    encoded = np.zeros((len(anchors), 4), dtype=float)

    for i, anchor in enumerate(anchors):

        if np.all(anchor == 0):
            continue

        x1_a, y1_a, x2_a, y2_a = anchor
        aw = x2_a - x1_a
        ah = y2_a - y1_a

        if aw <= 0 or ah <= 0:
            continue

        ax = (x1_a + x2_a) / 2
        ay = (y1_a + y2_a) / 2

        x1_g, y1_g, x2_g, y2_g = matched_boxes[i]
        gw = x2_g - x1_g
        gh = y2_g - y1_g

        gx = (x1_g + x2_g) / 2
        gy = (y1_g + y2_g) / 2

        # encode
        tx = (gx - ax) / aw
        ty = (gy - ay) / ah
        tw = np.log(max(gw / aw, 1e-6))
        th = np.log(max(gh / ah, 1e-6))

        encoded[i] = [tx, ty, tw, th]

    return encoded











