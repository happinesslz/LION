from scipy.optimize import linear_sum_assignment

import torch
from torch import nn

from pcdet.models.dense_heads.target_assigner.hungarian_assigner import overlaps
import copy

def box_cxcyczlwh_to_xyxyxy(x):
    x_c, y_c, z_c, l, w, h = x.unbind(-1)
    b = [
        (x_c - 0.5 * l),
        (y_c - 0.5 * w),
        (z_c - 0.5 * h),
        (x_c + 0.5 * l),
        (y_c + 0.5 * w),
        (z_c + 0.5 * h),
    ]

    return torch.stack(b, dim=-1)


def box_vol_wo_angle(boxes):
    vol = (boxes[:, 3] - boxes[:, 0]) * (boxes[:, 4] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 2])

    return vol


def box_intersect_wo_angle(boxes1, boxes2):
    ltb = torch.max(boxes1[:, None, :3], boxes2[:, :3])  # [N,M,3]
    rbf = torch.min(boxes1[:, None, 3:], boxes2[:, 3:])  # [N,M,3]

    lwh = (rbf - ltb).clamp(min=0)  # [N,M,3]
    inter = lwh[:, :, 0] * lwh[:, :, 1] * lwh[:, :, 2]  # [N,M]

    return inter


def box_iou_wo_angle(boxes1, boxes2):
    vol1 = box_vol_wo_angle(boxes1)
    vol2 = box_vol_wo_angle(boxes2)
    inter = box_intersect_wo_angle(boxes1, boxes2)

    union = vol1[:, None] + vol2 - inter
    iou = inter / union

    return iou, union


def generalized_box3d_iou(boxes1, boxes2):
    boxes1 = torch.nan_to_num(boxes1)
    boxes2 = torch.nan_to_num(boxes2)

    assert (boxes1[:, 3:] >= boxes1[:, :3]).all()
    assert (boxes2[:, 3:] >= boxes2[:, :3]).all()

    iou, union = box_iou_wo_angle(boxes1, boxes2)

    ltb = torch.min(boxes1[:, None, :3], boxes2[:, :3])  # [N,M,3]
    rbf = torch.max(boxes1[:, None, 3:], boxes2[:, 3:])  # [N,M,3]

    whl = (rbf - ltb).clamp(min=0)  # [N,M,3]
    vol = whl[:, :, 0] * whl[:, :, 1] * whl[:, :, 2]

    return iou - (vol - union) / vol


class HungarianMatcher3d(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, cost_rad: float = 1, \
                 decode_bbox_func=None, use_iou=False, iou_rectifier=[0.68, 0.71, 0.65], iou_cls=[0, 1, 2]):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_rad = cost_rad
        self.decode_bbox_func = decode_bbox_func
        self.iou_rectifier = iou_rectifier
        self.use_iou = use_iou
        self.iou_cls = iou_cls

        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0 or cost_rad != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        if "topk_indexes" in outputs.keys():
            pred_logits = torch.gather(
                outputs["pred_logits"],
                1,
                outputs["topk_indexes"].expand(-1, -1, outputs["pred_logits"].shape[-1]),
            )
            pred_boxes = torch.gather(
                outputs["pred_boxes"],
                1,
                outputs["topk_indexes"].expand(-1, -1, outputs["pred_boxes"].shape[-1]),
            )
            if self.use_iou and 'pred_ious' in outputs.keys():
                pred_iou = torch.gather(
                    outputs["pred_ious"],
                    1,
                    outputs["topk_indexes"].expand(-1, -1, outputs["pred_ious"].shape[-1]),
                )
        else:
            pred_logits = outputs["pred_logits"]
            pred_boxes = outputs["pred_boxes"]
            if self.use_iou and 'pred_ious' in outputs.keys():
                pred_iou = outputs["pred_ious"]

        bs, num_queries = pred_logits.shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = pred_logits.sigmoid()
        # ([batch_size, num_queries, 6], [batch_size, num_queries, 2])
        out_bbox = pred_boxes[..., :6]
        out_rad = pred_boxes[..., 6:7]

        if self.use_iou and 'pred_ious' in outputs.keys():
            out_iou = (pred_iou + 1) / 2

            iou_rectifier = self.iou_rectifier
            if isinstance(iou_rectifier, list):
                for i in range(len(iou_rectifier)):
                    if i not in self.iou_cls:
                        continue

                    out_prob[..., i] = torch.pow(out_prob[..., i], 1 - iou_rectifier[i]) * torch.pow(
                        out_iou[..., 0], iou_rectifier[i])
            elif isinstance(iou_rectifier, float):
                for i in range(out_prob.shape[-1]):
                    if i not in self.iou_cls:
                        continue
                    out_prob[..., i] = torch.pow(out_prob[..., i], 1 - iou_rectifier) * torch.pow(
                        out_iou[..., 0], iou_rectifier)
            else:
                raise TypeError('only list or float')


    # Also concat the target labels and boxes
        # [batch_size, num_target_boxes]
        tgt_ids = [v["labels"] for v in targets]
        # [batch_size, num_target_boxes, 6]
        tgt_bbox = [v["gt_boxes"][..., :6] for v in targets]
        # [batch_size, num_target_boxes, 2]
        tgt_rad = [v["gt_boxes"][..., 6:7] for v in targets]

        alpha = 0.25
        gamma = 2.0

        indices = []
        ious = []
        for i in range(bs):
            with torch.cuda.amp.autocast(enabled=False):
                out_prob_i = out_prob[i].float()
                out_bbox_i = out_bbox[i].float()
                out_rad_i = out_rad[i].float()
                tgt_bbox_i = tgt_bbox[i].float()
                tgt_rad_i = tgt_rad[i].float()

                # [num_queries, num_target_boxes]
                cost_giou = -generalized_box3d_iou(
                    box_cxcyczlwh_to_xyxyxy(out_bbox[i]),
                    box_cxcyczlwh_to_xyxyxy(tgt_bbox[i]),
                )

                neg_cost_class = (1 - alpha) * (out_prob_i ** gamma) * (-(1 - out_prob_i + 1e-8).log())
                pos_cost_class = alpha * ((1 - out_prob_i) ** gamma) * (-(out_prob_i + 1e-8).log())
                cost_class = pos_cost_class[:, tgt_ids[i]] - neg_cost_class[:, tgt_ids[i]]

                # Compute the L1 cost between boxes
                # [num_queries, num_target_boxes]
                cost_bbox = torch.cdist(out_bbox_i, tgt_bbox_i, p=1)
                cost_rad = torch.cdist(out_rad_i, tgt_rad_i, p=1)

            # Final cost matrix
            C_i = (
                    self.cost_bbox * cost_bbox
                    + self.cost_class * cost_class
                    + self.cost_giou * cost_giou
                    + self.cost_rad * cost_rad
            )
            # [num_queries, num_target_boxes]
            C_i = C_i.view(num_queries, -1).cpu()
            indice = linear_sum_assignment(C_i)
            indices.append(indice)

            if self.decode_bbox_func is not None:
                iou = overlaps(self.decode_bbox_func(copy.deepcopy(pred_boxes[i])), self.decode_bbox_func(copy.deepcopy(targets[i]['gt_boxes'])))
                mean_iou = iou[indice[0], indice[1]].sum() / max(len(indice[0]), 1)
                ious.append(float(mean_iou))

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices], ious

    def extra_repr(self):
        s = "cost_class={cost_class}, cost_bbox={cost_bbox}, cost_giou={cost_giou}, cost_rad={cost_rad}"

        return s.format(**self.__dict__)
