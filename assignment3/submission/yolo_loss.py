import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def compute_iou(box1, box2):
    """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    """
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(
        # [N,2] -> [N,1,2] -> [N,M,2]
        box1[:, :2].unsqueeze(1).expand(N, M, 2),
        # [M,2] -> [1,M,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),
    )

    rb = torch.min(
        # [N,2] -> [N,1,2] -> [N,M,2]
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),
        # [M,2] -> [1,M,2] -> [N,M,2]
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou


class YoloLoss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj, device):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj
        self.device = device

    def xywh2xyxy(self, boxes):
        """
        Parameters:
        boxes: (N,4) representing by x,y,w,h

        Returns:
        boxes: (N,4) representing by x1,y1,x2,y2

        if for a Box b the coordinates are represented by [x, y, w, h] then
        x1, y1 = x/S - 0.5*w, y/S - 0.5*h ; x2,y2 = x/S + 0.5*w, y/S + 0.5*h
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        """
        ### CODE ###
        x = boxes[:, 0]
        y = boxes[:, 1]
        half_width = boxes[:, 2] / 2
        half_height = boxes[:, 3] / 2
        x1 = x / self.S - half_width
        y1 = y / self.S - half_height
        x2 = x / self.S + half_width
        y2 = y / self.S + half_height

        # return torch.vstack((x1, y1, x2, y2))
        return torch.stack((x1, y1, x2, y2), dim=1)

    def find_best_iou_boxes(self, pred_box_list, box_target):
        """
        Parameters:
        box_pred_list : (list) [(tensor) size (-1, 5)]  
        box_target : (tensor)  size (-1, 4)

        Returns:
        best_iou: (tensor) size (-1, 1)
        best_boxes : (tensor) size (-1, 5), containing the boxes which give the best iou among the two (self.B) predictions

        Hints:
        1) Find the iou's of each of the 2 bounding boxes of each grid cell of each image.
        2) For finding iou's use the compute_iou function
        3) use xywh2xyxy to convert bbox format if necessary,
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        We perform this transformation to convert the correct coordinates into bounding box coordinates.
        """

        ### CODE ###
        # Your code here
        best_ious = torch.zeros(box_target.shape[0])
        best_boxes = torch.zeros_like(pred_box_list[0])
        box_target = self.xywh2xyxy(box_target)
        ious = torch.zeros(box_target.shape[0], self.B)

        for box in range(len(pred_box_list)):
            ious[:, box] = compute_iou(
                self.xywh2xyxy(pred_box_list[box][:, :4]),
                box_target
            ).diag()
        for i in range(box_target.shape[0]):
            iou = ious[i]
            best_boxes[i] = pred_box_list[torch.argmax(iou)][i]
            best_ious[i] = torch.max(iou)

        return best_ious.unsqueeze(1).detach().to(self.device), \
            best_boxes.to(self.device)

    def get_class_prediction_loss(self, classes_pred, classes_target, has_object_map):
        """
        Parameters:
        classes_pred : (tensor) size (batch_size, S, S, 20)
        classes_target : (tensor) size (batch_size, S, S, 20)
        has_object_map: (tensor) size (batch_size, S, S)

        Returns:
        class_loss : scalar
        """
        ### CODE ###
        # Your code here
        ans = (has_object_map *
               (classes_pred - classes_target).pow(2).sum(dim=-1)).sum()
        # print(f"ans shape: {ans.shape}")
        return ans

    def get_no_object_loss(self, pred_boxes_list, has_object_map):
        """
        Parameters:
        pred_boxes_list: (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        has_object_map: (tensor) size (N, S, S)

        Returns:
        loss : scalar

        Hints:
        1) Only compute loss for cell which doesn't contain object
        2) compute loss for all predictions in the pred_boxes_list list
        3) You can assume the ground truth confidence of non-object cells is 0
        """
        ### CODE ###
        loss = 0
        for box in range(len(pred_boxes_list)):
            loss += F.mse_loss(
                pred_boxes_list[box][:, :, :,
                                     4][has_object_map == 0],
                torch.zeros_like(
                    pred_boxes_list[box][:, :, :, 4][has_object_map == 0]),
                reduction="sum"
            )

        return loss

    def get_contain_conf_loss(self, box_pred_conf, box_target_conf):
        """
        Parameters:
        box_pred_conf : (tensor) size (-1,1)
        box_target_conf: (tensor) size (-1,1)

        Returns:
        contain_loss : scalar

        Hints:
        The box_target_conf should be treated as ground truth, i.e., no gradient

        """
        # CODE
        # your code here
        return F.mse_loss(box_pred_conf, box_target_conf, reduction="sum")

    def get_regression_loss(self, box_pred_response, box_target_response):
        """
        Parameters:
        box_pred_response : (tensor) size (-1, 4)
        box_target_response : (tensor) size (-1, 4)
        Note : -1 corresponds to ravels the tensor into the dimension specified
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as

        Returns:
        reg_loss : scalar

        """
        # CODE
        # your code here
        return (F.mse_loss(
            box_pred_response[:, :2].sqrt(),
            box_target_response[:, :2].sqrt(), reduction="sum"
        ) +
            F.mse_loss(
            box_pred_response[:, 2:4].sqrt(),
            box_target_response[:, 2:4].sqrt(), reduction="sum")
        )

    def forward(self, pred_tensor, target_boxes, target_cls, has_object_map):
        """
        pred_tensor: (tensor) size(N,S,S,Bx5+20=30) where:  
                            N - batch_size
                            S - width/height of network output grid
                            B - number of bounding boxes this grid cell is a part of = 2
                            5 - number of bounding box values corresponding to [x, y, w, h, c]
                                where x - x_coord, y - y_coord, w - width, h - height, c - confidence of having an object
                            20 - number of classes

        target_boxes: (tensor) size (N, S, S, 4): the ground truth bounding boxes
        target_cls: (tensor) size (N, S, S, 20): the ground truth class
        has_object_map: (tensor, bool) size (N, S, S): the ground truth for whether each cell contains an object (True/False)

        Returns:
        loss_dict (dict): with key value stored for total_loss, reg_loss, containing_obj_loss, no_obj_loss and cls_loss
        """
        N = pred_tensor.size(0)
        total_loss = 0.0

        # split the pred tensor from an entity to separate tensors:
        # -- pred_boxes_list: a list containing all bbox prediction (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        # -- pred_cls (containing all classification prediction)
        # your code here
        pred_boxes_list = torch.stack([
            pred_tensor[:, :, :, box * 5: (box + 1) * 5] for box in range(self.B)
        ])
        pred_cls = pred_tensor[:, :, :, self.B * 5:]

        # compcute classification loss
        # your code here
        # print("computing prediction loss...")
        classification_loss = self.get_class_prediction_loss(
            pred_cls, target_cls, has_object_map)

        # compute no-object loss
        # your code here
        # print("computing noobj loss...")
        no_obj_loss = self.get_no_object_loss(pred_boxes_list, has_object_map) \
            * self.l_noobj

        # Re-shape boxes in pred_boxes_list and target_boxes to meet the following desires
        # 1) only keep having-object cells
        # 2) vectorize all dimensions except for the last one for faster computation
        # your code here
        pred_boxes_list = [x[has_object_map] for x in pred_boxes_list]
        target_boxes = target_boxes[has_object_map]

        # find the best boxes among the 2 (or self.B) predicted boxes and the corresponding iou
        # your code here

        # print("computing best iou boxes...")
        best_ious, best_boxes = self.find_best_iou_boxes(
            pred_boxes_list, target_boxes)

        # compute regression loss between the found best bbox and GT bbox for all the cell containing objects
        # your code here

        # print("computing regression_loss...")
        regression_loss = self.get_regression_loss(
            best_boxes[:, :4], target_boxes) * self.l_coord

        # compute contain_object_loss
        # your code here
        # print("computing contain_object_loss...")
        contain_object_loss = self.get_contain_conf_loss(
            best_boxes[:, 4].view(*(best_boxes[:, 4].shape), -1), best_ious)

        # compute final loss
        # your code here
        total_loss += (classification_loss + no_obj_loss +
                       contain_object_loss + regression_loss) / N

        # print(
        #     total_loss.shape,
        #     regression_loss.shape,
        #     contain_object_loss.shape,
        #     no_obj_loss.shape,
        #     classification_loss.shape
        # )
        # construct return loss_dict
        # your code for loss_dict here
        loss_dict = dict(
            total_loss=total_loss,
            reg_loss=regression_loss,
            containing_obj_loss=contain_object_loss,
            no_obj_loss=no_obj_loss,
            cls_loss=classification_loss
        )
        return loss_dict
