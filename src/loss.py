import torch
import torch.nn as nn
import util

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
        iou_b1 = util.intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = util.intersection_over_union(predictions[..., 26:30], target[..., 21:25])
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
