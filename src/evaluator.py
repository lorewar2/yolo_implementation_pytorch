import torch
import torchvision.transforms as transforms
from model import Custom_yolo
from dataset import VOCDataset
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from loss import Custom_loss_function
import util

IMAGE_DIR = "data/images"
LABEL_DIR = "data/labels"
MODEL_SAVE_PATH = "model/saved_model.pt"

class Compose (object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, boundingboxes):
        for t in self.transforms:
            image, boundingboxes = t(image), boundingboxes
        return image, boundingboxes

def evaluator():
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
    model = torch.load(MODEL_SAVE_PATH)
    model.eval()
    # get the predicted boxes and target boxes
    pred_boxes, target_boxes = util.bounding_box_calculator(train_loader, model, iou_threshold = 0.5, threshold = 0.1)
    #print(pred_boxes)
    return pred_boxes, target_boxes

def compute_ap(recalls, precisions):
    # calculates the ap using recall and precision
    mrec = np.concatenate(([0.], recalls, [1.]))
    mpre = np.concatenate(([0.], precisions, [0.]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = sum((mrec[i + 1] - mrec[i]) * mpre[i + 1] for i in range(len(mpre) - 1))
    return ap

def evaluate_predictions_per_class(predictions, ground_truths, class_id, iou_threshold=0.5):
    # get the boxes in this class
    pred_boxes = [p for p in predictions if int(p[1]) == class_id]
    gt_boxes = [gt for gt in ground_truths if int(gt[1]) == class_id]
    print(len(gt_boxes))
    print(len(pred_boxes))
    assigned_gt = [False] * len(gt_boxes)
    tp = []
    fp = []
    # go through the predboxes and if the pred box has a iou greater than 0.5 with a groundtruth box label as true positive
    for pred_box in pred_boxes:
        matched = False
        for i, gt_box in enumerate(gt_boxes):
            if util.intersection_over_union(torch.tensor(gt_box[3:]), torch.tensor(pred_box[3:]),) > iou_threshold:
                if not assigned_gt[i]:
                    assigned_gt[i] = True
                    tp.append(1)
                    fp.append(0)
                    matched = True
                    break
        if not matched:
            tp.append(0)
            fp.append(1)
    # calculate recall and precision for this class
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recalls = tp_cumsum / (len(gt_boxes) if len(gt_boxes) else 1)
    precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(np.float64).eps)
    # calulate ap using recall and precision and return
    return compute_ap(recalls, precisions)

def evaluate_predictions(predictions, ground_truths, num_classes = 20):
    # compute mAP for 20 classes
    aps = []
    for class_id in range(1, num_classes + 1):
        ap = evaluate_predictions_per_class(predictions, ground_truths, class_id)
        aps.append(ap)
    print(aps)
    mean_ap = np.mean(aps)
    return mean_ap