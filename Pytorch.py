import warnings
warnings.filterwarnings("ignore", message="Failed to load image Python extension.*")
import torch
import time
import os
import glob
import cv2
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.optim as optim
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDClassificationHead
import torch.mps

IMG_SIZE = 300


def hand_collate_fn(batch):

    return tuple(zip(*batch))


def get_ssd_model(num_classes=1):

    model = ssd300_vgg16(weights=None)

    anchor_generator = model.anchor_generator
    num_anchors = anchor_generator.num_anchors_per_location()
    dummy_input = torch.rand(1, 3, IMG_SIZE, IMG_SIZE)
    backbone_outputs = model.backbone(dummy_input)
    in_channels = [feature.shape[1] for feature in backbone_outputs.values()]

    model.head.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes)

    return model


class BoundingBox:
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin, self.ymin, self.xmax, self.ymax = xmin, ymin, xmax, ymax


    def to_tensor(self, orig_w, orig_h):
        scale_x = IMG_SIZE / orig_w
        scale_y = IMG_SIZE / orig_h

        return torch.tensor([
            self.xmin * scale_x,
            self.ymin * scale_y,
            self.xmax * scale_x,
            self.ymax * scale_y
        ], dtype=torch.float32)


def parse_annotations(xml_path, img_dir):

    tree = ET.parse(xml_path)
    root = tree.getroot()
    fname = (root.findtext('filename') or root.findtext('path').split(os.sep)[-1])
    img_path = os.path.join(img_dir, fname)
    boxes = []

    for obj in root.findall('object'):
        bnd = obj.find('bndbox')
        xmin = int(bnd.findtext('xmin'))
        ymin = int(bnd.findtext('ymin'))
        xmax = int(bnd.findtext('xmax'))
        ymax = int(bnd.findtext('ymax'))
        boxes.append(BoundingBox(xmin, ymin, xmax, ymax))
    return img_path, boxes


class HandDataset(Dataset):

    def __init__(self, xml_dir, img_dir, bg_dir=None, transforms=None):

        self.transforms = transforms
        self.samples = []

        for xml in sorted(glob.glob(os.path.join(xml_dir, '*.xml'))):
            img_path, boxes = parse_annotations(xml, img_dir)
            if boxes:
                self.samples.append((img_path, boxes))

        if bg_dir:
            bg_images = glob.glob(os.path.join(bg_dir, '*.jpg')) + \
                        glob.glob(os.path.join(bg_dir, '*.png'))
            for bg_img in bg_images:
                self.samples.append((bg_img, []))

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        img_path, boxes = self.samples[idx]
        img = cv2.imread(img_path)

        if img is None:
            raise RuntimeError(f"Failed to load {img_path}")

        h, w = 1080, 1920
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2,0,1).float().div(255.0)

        if self.transforms:
            img = self.transforms(img)

        if boxes:
            boxes_t = torch.stack([b.to_tensor(w, h) for b in boxes], 0)
            labels = torch.ones((len(boxes),), dtype=torch.int64)  # class 1 = hand
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {'boxes': boxes_t, 'labels': labels}
        return img, target


def get_hand_dataloader(xml_dir, img_dir, bg_dir,
                        batch_size=8, transforms=None,
                        shuffle=True, num_workers=0):

    ds = HandDataset(xml_dir, img_dir, bg_dir, transforms)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=hand_collate_fn
    )


def draw_boxes(img_tensor, boxes, scores=None, color=(0, 255, 0)):

    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.permute(1, 2, 0).cpu().numpy() * 255
        img = img.astype("uint8").copy()
    else:
        img = img_tensor.copy()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.int().tolist()
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if scores is not None:
            score = scores[i].item()
            cv2.putText(img, f"{score:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return img


def convert_to_original_scale(box, original_shape):

    og_w = original_shape[1]
    og_h = original_shape[0]
    scale_x = og_w / IMG_SIZE
    scale_y = og_h / IMG_SIZE
    x1, y1, x2, y2 = box
    x1 *= scale_x
    x2 *= scale_x
    y1 *= scale_y
    y2 *= scale_y
    return int(x1), int(y1), int(x2), int(y2)


def preprocess(img):

    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return img


def decode_predictions(preds, conf_thresh=0.5):

    pred = preds[0]
    boxes = pred["boxes"]
    scores = pred["scores"]
    labels = pred["labels"]

    keep = scores > conf_thresh
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    return boxes, scores, labels


def live_feed(model, device):

    model.load_state_dict(torch.load('Last_model_weights.pth'))
    model.eval()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(1)


    if not cap.isOpened():
        raise IOError("No camera found")
        
    model.to(device)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_img = cv2.resize(frame, (300, 300))
        input_tensor = preprocess(input_img).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(input_tensor)
        # Decode predictions from model output
        boxes, scores, labels = decode_predictions(preds, conf_thresh=0.5)

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = convert_to_original_scale(box, original_shape=frame.shape)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            cv2.putText(frame, f"{score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Live Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ── USAGE ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = get_ssd_model(num_classes=2).to(device)

    train_or_test = input("Do You Want To Train Or Test?: ")

    if train_or_test == "train" or train_or_test == "Train":
        batch = 8
        train_loader = get_hand_dataloader(
            xml_dir="/Users/bronsonf/Desktop/Coding/Hand Dataset/XML",
            img_dir="/Users/bronsonf/Desktop/Coding/Hand Dataset/Hand",
            bg_dir="/Users/bronsonf/Desktop/Coding/Hand Dataset/background",
            batch_size=batch,
            transforms=None,
            shuffle=True,
            num_workers=4
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=.1e-4, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

        model.train()
        for epoch in range(20):

            start = time.time()
            for images, targets in train_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

            end = time.time()
            elapsed_time = end - start
            print(f"Epoch {epoch + 1} loss: {loss.item():.4f} and took {elapsed_time / 60: .2f} min")

        torch.save(model.state_dict(), 'model_weights.pth')
        print("Model Weights have been saved to: 'model_weights.pth'")

    elif train_or_test == "test" or train_or_test == "Test":
        live_feed(model, device)

    else:
        print("Pick 'train' or 'test' Next Time")


def get_prediction(model, device, frame, conf_thresh=0.5):

    IMG_SIZE = 300
    # Resize frame to model input size
    input_img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    input_tensor = preprocess(input_img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        preds = model(input_tensor)

    boxes, scores, labels = decode_predictions(preds, conf_thresh=conf_thresh)

    if len(boxes) > 0:
        # Convert first box back to original frame scale
        x1, y1, x2, y2 = convert_to_original_scale(boxes[0], original_shape=frame.shape)
        return [x1, y1, x2, y2]
    else:
        return None
