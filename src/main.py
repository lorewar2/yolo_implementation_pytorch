import torch
import torchvision.transforms as transforms
from model import Custom_yolo
from dataset import VOCDataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
import torch.optim as optim
from loss import Custom_loss_function

IMAGE_DIR = "data/images"
LABEL_DIR = "data/labels"

class Compose (object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, boundingboxes):
        for t in self.transforms:
            image, boundingboxes = t(image), boundingboxes
        return image, boundingboxes

def main ():
    # preprocess the data, transform
    transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])
    train_dataset = VOCDataset (
        "data/100examples.csv",
        transform = transform,
        img_dir = IMAGE_DIR,
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
    model = Custom_yolo()
    # train the model
    # parameters
    epochs = 500
    optimizer = optim.Adam(model.parameters(), lr = 0.00002, weight_decay = 0)
    mean_loss = []
    cus_loss = Custom_loss_function()
    # train loop
    for epoch in range(epochs):
        print(mean_loss)
        for batch_idx, (x, y) in enumerate(train_loader):
            out = model(x)
            loss = cus_loss(out, y)
            mean_loss.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("Training: epoch: {} batch: {}/7".format(epoch, batch_idx))
    # get the predicted boxes and target boxes
    pred_boxes, target_boxes = bounding_box_calculator(train_loader, model, iou_threshold = 0.1, threshold = 0.1)

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
                if box[2] > 0.15:
                    idx_pred_boxes.append(box)
                    print(get_class_name(box[1]))
        plot_both_images_with_boxes(train_dataset[idx][0], idx_target_boxes, idx_pred_boxes)
    return

def get_class_name(value):
    # from the value get the string of the class
    name_list = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    if value >= 19.0:
        return ""
    else:
        return name_list[int(value)]

def plot_both_images_with_boxes (image, true_boxes, pred_boxes):
    im = np.array(image)
    im = np.transpose(im, (1, 2, 0))
    height, width, x = im.shape
    _, ax = plt.subplots(2)
    # the truth labelled image in red
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
    # the predicted image in blue
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
            edgecolor = "b",
            facecolor = "none",
        )
        ax[1].add_patch(rect)
    plt.show()

def threshold_check (boundingboxes, threshold_2, threshold_1):
    # get the bounding boxes which are greater than threshold_1
    boundingboxes = [box for box in boundingboxes if box[1] > threshold_1]
    # sort the bounding boxes
    boundingboxes = sorted(boundingboxes, key = lambda x: x[1], reverse = True)
    # get the bounding boxes which are greater than threshold_2
    boundingboxes_after_nms = []
    while boundingboxes:
        chosen_box = boundingboxes.pop(0)
        boundingboxes = [box for box in boundingboxes if (box[0] != chosen_box[0] or intersection_over_union(torch.tensor(chosen_box[2:]), torch.tensor(box[2:]),) < threshold_2)]
        boundingboxes_after_nms.append(chosen_box) 
    return boundingboxes_after_nms

def intersection_over_union(boxes_preds, boxes_labels):
    # calculate the top left corner of the predicted bounding boxes
    b1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
    b1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
    # calculate the bottom right corner of the predicted bounding boxes
    b1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
    b1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
    # calculate the top left corner of the ground truth bounding boxes
    b2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
    b2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
    # calculate the bottom right corner of the ground truth bounding boxes
    b2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
    b2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
    # calculate the coordinates of the intersection rectangle's top left corner
    x1 = torch.max(b1_x1, b2_x1)
    y1 = torch.max(b1_y1, b2_y1)
    # calculate the coordinates of the intersection rectangle's bottom right corner
    x2 = torch.min(b1_x2, b2_x2)
    y2 = torch.min(b1_y2, b2_y2)
    # calculate the area of the predicted bounding boxes
    b1_area = abs((b1_x2 - b1_x1) * (b1_y2 - b1_y1))
    # calculate the area of the ground truth bounding boxes
    b2_area = abs((b2_x2 - b2_x1) * (b2_y2 - b2_y1))
    # calculate the area of the intersection rectangle
    b1_b2_intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    # calculate the denominator for the IoU formula, adjusting for potential zero division
    denominator = (b1_area + b2_area - b1_b2_intersection + 1e-5)
    # calculate and return the IoU value
    return b1_b2_intersection / denominator

def bounding_box_calculator (loader, model, iou_threshold, threshold):
    all_pred_boxes = []
    all_true_boxes = []
    # put the model in eval mode
    model.eval()
    train_idx = 0
    # go through the images 
    for (x, labels) in (loader):
        with torch.no_grad():
            predictions = model(x)
        batch_size = x.shape[0]
        true_boundingboxes = cellboxes_to_boxes(labels)
        boundingboxes = cellboxes_to_boxes(predictions)
        for idx in range(batch_size):
            # get the boxes which passes the threshold check
            nms_boxes = threshold_check(boundingboxes[idx], threshold_2 = iou_threshold, threshold_1 = threshold)
            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)
            for box in true_boundingboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)
            train_idx += 1
    model.train()
    return all_pred_boxes, all_true_boxes

def cellboxes_to_boxes (input):
    batch_size = input.shape[0]
    predictions = input.reshape(batch_size, 7, 7, 30)
    boundingboxes1 = predictions[..., 21:25]
    boundingboxes2 = predictions[..., 26:30]
    scores = torch.cat((predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim = 0)
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = boundingboxes1 * (1 - best_box) + best_box * boundingboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / 7 * (best_boxes[..., :1] + cell_indices)
    y = 1 / 7 * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / 7 * best_boxes[..., 2:4]
    converted_boundingboxes = torch.cat((x, y, w_y), dim = -1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(-1)
    converted_pred = torch.cat((predicted_class, best_confidence, converted_boundingboxes), dim=-1).reshape(input.shape[0], 7 * 7, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_boundingboxes = []
    for ex_idx in range(input.shape[0]):
        boundingboxes = []
        for boundingboxes_idx in range(7 * 7):
            boundingboxes.append([x.item() for x in converted_pred[ex_idx, boundingboxes_idx, :]])
        all_boundingboxes.append(boundingboxes)
    return all_boundingboxes

if __name__ == "__main__":
    main()
   