import torch
import torch.nn as nn

class Custom_loss_function(nn.Module):
    def __init__(self):
        super(Custom_loss_function, self).__init__()
        self.mse = nn.MSELoss(reduction = "sum")
        self.S = 7 # split size
        self.B = 2 # number of boxes
        self.C = 20 # number of classes
        # no object loss modifer
        self.lambda_no_object = 0.5
        # box loss modifier
        self.lambda_coordinate = 5

    def forward(self, predictions, target):
        # calculate the required data
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, bestbox = torch.max(ious, dim=0) 
        exists_box = target[..., 20].unsqueeze(3)
        box_predictions = exists_box * ((bestbox * predictions[..., 26:30]+ (1 - bestbox) * predictions[..., 21:25]))
        box_targets = exists_box * target[..., 21:25]
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])  
        # box loss
        box_loss = self.mse(torch.flatten(box_predictions, end_dim = -2),torch.flatten(box_targets, end_dim = -2))
        # prediction loss
        pred_box = (bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21])
        # loss when there is a object
        object_loss = self.mse(torch.flatten(exists_box * pred_box), torch.flatten(exists_box * target[..., 20:21] * iou_maxes))
        # loss when there is no object
        no_object_loss = self.mse(torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim = 1), torch.flatten((1 - exists_box) * target[..., 20:21], start_dim = 1))
        no_object_loss += self.mse(torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim = 1), torch.flatten((1 - exists_box) * target[..., 20:21], start_dim = 1))
        # class prediction loss
        class_loss = self.mse(torch.flatten(exists_box * predictions[..., :20], end_dim = -2), torch.flatten(exists_box * target[..., :20], end_dim = -2))
        # aggregate all
        loss = (self.lambda_coordinate * box_loss + object_loss + self.lambda_no_object * no_object_loss + class_loss)
        return loss

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