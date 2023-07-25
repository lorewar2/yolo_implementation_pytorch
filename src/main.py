import torch
import torchvision.transforms as transforms
from model import Yolo
from dataset import VOCDataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
import torch.optim as optim
from loss import YoloLoss

IMG_DIR = "data/images"
LABEL_DIR = "data/labels"

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes

def main():
    # preprocess the data, transform
    transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])
    train_dataset = VOCDataset (
        "data/100examples.csv",
        transform = transform,
        img_dir = IMG_DIR,
        label_dir = LABEL_DIR
    )
    # make the data loader to load the data
    train_loader = DataLoader (
        dataset = train_dataset,
        batch_size = 16,
        shuffle = False,
        drop_last = True
    )
    # make the model
    model = Yolo()
    # train the model
    epochs = 300
    optimizer = optim.Adam(model.parameters(), lr = 0.00002, weight_decay = 0)
    mean_loss = []
    test = YoloLoss()
    for epoch in range(epochs):
        print(mean_loss)
        for batch_idx, (x, y) in enumerate(train_loader):
            out = model(x)
            loss = test(out, y)
            mean_loss.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("Training: epoch: {} batch: {}/7".format(epoch, batch_idx))
    # get the predicted boxes and target boxes
    pred_boxes, target_boxes = get_bboxes(train_loader, model, iou_threshold = 0, threshold = 0)

    # draw each image for 
    for idx in range(100):
        idx_target_boxes = []
        idx_pred_boxes = []
        print("Target items:")
        for box in target_boxes:
            if box[0] == idx:
                idx_target_boxes.append(box)
                print(get_class_name(box[1]))
        print("Pred items:")
        for box in pred_boxes:
            if box[0] == idx:
                if box[2] > 0.05:
                    idx_pred_boxes.append(box)
                    print(get_class_name(box[1]))
        plot_both_images_with_boxes(train_dataset[idx][0], idx_target_boxes, idx_pred_boxes)
    return

def get_class_name(x):
    name_list = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    if x >= 19.0:
        return ""
    return name_list[int(x)]

def plot_both_images_with_boxes(image, true_boxes, pred_boxes):
    im = np.array(image)
    im = np.transpose(im, (1, 2, 0))
    height, width, x = im.shape
    _, ax = plt.subplots(2)
    ax[0].imshow(im, interpolation='nearest')
    for box in true_boxes:
        box = box[3:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth = 1,
            edgecolor = "r",
            facecolor = "none",
        )
        ax[0].add_patch(rect)
    ax[1].imshow(im, interpolation='nearest')
    for box in pred_boxes:
        box = box[3:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth = 2,
            edgecolor = "r",
            facecolor = "none",
        )
        ax[1].add_patch(rect)
    plt.show()

def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    assert type(bboxes) == list
    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []
    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold 
        ]
        bboxes_after_nms.append(chosen_box) 
    return bboxes_after_nms

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    return intersection / (box1_area + box2_area - intersection + 1e-6)

def get_bboxes(loader, model, iou_threshold, threshold, pred_format="cells", box_format="midpoint"):
    all_pred_boxes = []
    all_true_boxes = []
    model.eval()
    train_idx = 0
    for (x, labels) in (loader):
        with torch.no_grad():
            predictions = model(x)
        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)
        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )
            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)
            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1
    model.train()
    return all_pred_boxes, all_true_boxes

def convert_cellboxes(predictions, S=7):
    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )
    return converted_preds

def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []
    for ex_idx in range(out.shape[0]):
        bboxes = []
        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)
    return all_bboxes

if __name__ == "__main__":
    main()
   