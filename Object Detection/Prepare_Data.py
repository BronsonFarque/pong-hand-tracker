import cv2
import os
import numpy as np
from sklearn.datasets import fetch_openml
import glob
import xml.etree.ElementTree as ET
import pickle


IMG_SIZE = 224
NUM_CLASSES = 1  # Only 'hand' class


class BoundingBox:

    def __init__(self, img_path, xmin, ymin, xmax, ymax):

        self.img_path = img_path
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.bbox = [xmin, ymin, xmax, ymax]

    def normalize(self, img_w, img_h):

        x_scale = 224 / img_w
        y_scale = 224 / img_h
        return [
            self.xmin * x_scale,
            self.ymin * y_scale,
            self.xmax * x_scale,
            self.ymax * y_scale
        ]


class AnchorBox:
    def __init__(self, scale, size, ratio):

        self.height = scale * size
        self.width = self.height * ratio

    def __repr__(self):

        return f"AnchorBox(h={self.height}, w={self.width})"


def generate_anchor_boxes(feature_map_shapes, base_sizes, ratios, scales):
    """
    feature_map_shapes: List of tuples (h, w) for each SSD feature map
    base_sizes: List of ints, each base size corresponding to a feature map
    ratios: List of ratios to use for all feature maps (or per-layer if needed)
    scales: List of scales to use for all feature maps (or per-layer if needed)
    """
    all_anchors = []

    for fmap_idx, (grid_h, grid_w) in enumerate(feature_map_shapes):
        stride = base_sizes[fmap_idx]

        for i in range(grid_h):
            for j in range(grid_w):
                center_x = (j + 0.5) * stride
                center_y = (i + 0.5) * stride

                for scale in scales:
                    for ratio in ratios:
                        width = scale * stride * np.sqrt(ratio)
                        height = scale * stride / np.sqrt(ratio)

                        xmin = center_x - width / 2
                        ymin = center_y - height / 2
                        xmax = center_x + width / 2
                        ymax = center_y + height / 2

                        all_anchors.append([xmin, ymin, xmax, ymax])

    return np.array(all_anchors)



def load_and_resize_image(path):

    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image at {path}")
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE))


def parse_annotations(xml_path):

    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = os.path.basename(root.find('path').text)
    img_path = os.path.join('/Users/bronsonf/Desktop/Coding/Hand Dataset/Hand', filename)
    bboxes = []

    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        bboxes.append(BoundingBox(img_path, xmin, ymin, xmax, ymax))
    return bboxes


def create_hand_data(xml_dir, img_dir):

    image_data = []
    label_data = []
    all_xml = sorted(glob.glob(os.path.join(xml_dir, '*.xml')))

    for xml_file in all_xml:
        bboxes = parse_annotations(xml_file)
        if not bboxes:
            continue

        img_file = bboxes[0].img_path
        img = load_and_resize_image(img_file)
        h, w = 1080, 1920

        image_data.append(img)
        norm_bboxes = [[*bbox.normalize(w, h), 0] for bbox in bboxes]  # class_id = 0
        label_data.append(norm_bboxes)

    image_data = np.array(image_data)
    with open("hand_data.pkl", "wb") as f:
        pickle.dump((image_data, label_data), f)

    return image_data, label_data


def split_data(image_data, label_data, train_ratio=0.8):

    total = len(image_data)
    split_idx = int(train_ratio * total)
    X_train, X_test = image_data[:split_idx], image_data[split_idx:]
    y_train, y_test = label_data[:split_idx], label_data[split_idx:]
    np.save("X_train.npy", X_train)
    np.save("y_train.npy", y_train)
    np.save("X_test.npy", X_test)
    np.save("y_test.npy", y_test)

if __name__ == "__main__":

    xml_dir = '/Users/bronsonf/Desktop/Coding/Hand Dataset/XML'
    img_dir = '/Users/bronsonf/Desktop/Coding/Hand Dataset/Hand'
    images, labels = create_hand_data(xml_dir, img_dir)
    split_data(images, labels)
    feature_map_shapes = [(28, 28), (14, 14), (7, 7)]
    base_sizes = [8, 16, 32]
    ratios = [.7, .6, .5, .4]
    scales = [2.1, 1.8, 1.5, 1.2]
    anchors = generate_anchor_boxes(feature_map_shapes, base_sizes, ratios, scales)
    np.save("anchor_boxes.npy", anchors)
    print(f"Prepared {len(images)} samples and {len(anchors)} anchor boxes.")


def load_mnist_sklearn(num_samples):

    mnist = fetch_openml('mnist_784', version=1, cache=True)
    x_data = mnist.data.to_numpy().astype(np.float32) / 255.0
    y_data = np.eye(10)[mnist.target.astype(int)]

    return x_data[:num_samples].reshape(-1, 1, 28, 28), y_data[:num_samples]


def load_fashion_mnist_sklearn(num_samples):

    fashion_mnist = fetch_openml('Fashion-MNIST', version=1, cache=True)
    x_data = fashion_mnist.data.to_numpy().astype(np.float32) / 255.0
    y_data = np.eye(10)[fashion_mnist.target.astype(int)]  # One-hot encoding

    return x_data[:num_samples].reshape(-1, 1, 28, 28), y_data[:num_samples]


def load_cifar10_sklearn(num_samples):

    cifar10 = fetch_openml('cifar_10', version=1, cache=True)
    x_data = cifar10.data.to_numpy().astype(np.float32) / 255.0
    y_data = np.eye(10)[cifar10.target.astype(int)]

    return x_data[:num_samples].reshape(-1, 3, 32, 32), y_data[:num_samples]

