import copy
import numpy as np

import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from ..model_utils import centernet_utils
from ..model_utils import model_nms_utils
from ...utils import loss_utils
from ...utils.spconv_utils import spconv


class SeparateHead(nn.Module):

    def __init__(self, input_channels, sep_head_dict, conv_type='spconv', bias_before_bn=False):
        super().__init__()

        conv_cls = {'subm': spconv.SubMConv2d, 'spconv': spconv.SparseConv2d}[conv_type]

        self.sep_head_dict = sep_head_dict
        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(spconv.SparseSequential(
                    conv_cls(input_channels, input_channels, 3, 1, 1, bias=bias_before_bn, indice_key=cur_name),
                    nn.BatchNorm1d(input_channels),
                    nn.ReLU()
                ))
            fc_list.append(spconv.SubMConv2d(input_channels, output_channels, 1, bias=True, indice_key=cur_name+'out'))
            fc = nn.Sequential(*fc_list)

            for m in fc.modules():
                if isinstance(m, (spconv.SubMConv2d, spconv.SparseConv2d)):
                    kaiming_normal_(m.weight.data)
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.constant_(m.bias, 0)

            if 'heatmap' == cur_name:
                fc[-1].bias.data.fill_(-2.19)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)
        return ret_dict


class SparseCenterHead(nn.Module):

    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_class = num_class
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.feature_map_stride = model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)
        self.use_nearestmap = model_cfg.TARGET_ASSIGNER_CONFIG.get('USE_NEARESTMAP', False)

        self.bn_momentum = model_cfg.get('BN_MOM', 0.1)
        self.bn_eps = model_cfg.get('BN_EPS', 1e-5)

        self.class_names = class_names
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []

        for cur_class_names in model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in class_names]
            )).cuda()
            self.class_id_mapping_each_head.append(cur_class_id_mapping)

        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'

        self.heads_list = nn.ModuleList()
        self.separate_head_cfg = model_cfg.SEPARATE_HEAD_CFG
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            cur_head_dict = copy.deepcopy(self.separate_head_cfg.HEAD_DICT)
            cur_head_dict['heatmap'] = dict(out_channels=len(cur_class_names), num_conv=model_cfg.NUM_HM_CONV)
            self.heads_list.append(
                SeparateHead(
                    input_channels=input_channels,
                    sep_head_dict=cur_head_dict,
                    conv_type=model_cfg.get('HEAD_CONV_TYPE', 'spconv'),
                    bias_before_bn=model_cfg.get('USE_BIAS_BEFORE_NORM', False)
                )
            )
        self.forward_ret_dict = {}
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.momentum = self.bn_momentum
                m.eps = self.bn_eps

    def assign_target_of_single_head(
            self, num_classes, gt_boxes, feature_map_size, feature_map_stride, spatial_indices,
            num_max_objs=500, gaussian_overlap=0.1, min_radius=2):

        # filter invalid gt boxes
        gt_boxes = gt_boxes[(gt_boxes[:, 3] > 0) & (gt_boxes[:, 4] > 0)]
        gt_boxes = gt_boxes[:num_max_objs]

        # center
        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)
        coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)
        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()

        # radius
        dx, dy = gt_boxes[:, 3], gt_boxes[:, 4]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride
        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        # compute training targets
        heatmap = gt_boxes.new_zeros(num_classes, len(spatial_indices))
        ret_boxes = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] - 1 + 1))
        inds = gt_boxes.new_zeros(num_max_objs).long()
        mask = gt_boxes.new_zeros(num_max_objs).bool()
        ret_boxes_src = gt_boxes.new_zeros(num_max_objs, gt_boxes.shape[-1])
        ret_boxes_src[:gt_boxes.shape[0]] = gt_boxes

        for k in range(gt_boxes.shape[0]):
            cur_class_id = (gt_boxes[k, -1] - 1).long()
            center_distances = ((spatial_indices - center_int[k][None])**2).sum(-1)
            centernet_utils.draw_gaussian_to_sparse_heatmap(
                heatmap[cur_class_id], center_distances, radius[k].item(), normalize=True)

            inds[k] = center_distances.argmin()
            mask[k] = True

        valid_num = gt_boxes.shape[0]
        ret_boxes[:valid_num, 0:2] = center[:valid_num] - spatial_indices[inds[:valid_num]]
        ret_boxes[:valid_num, 2] = z[:valid_num]
        ret_boxes[:valid_num, 3:6] = gt_boxes[:valid_num, 3:6].log()
        ret_boxes[:valid_num, 6] = torch.cos(gt_boxes[:valid_num, 6])
        ret_boxes[:valid_num, 7] = torch.sin(gt_boxes[:valid_num, 6])
        if gt_boxes.shape[1] > 8:
            ret_boxes[:valid_num, 8:] = gt_boxes[:valid_num, 7:]

        return heatmap, ret_boxes, inds, mask, ret_boxes_src

    def assign_targets(self, gt_boxes, feature_map_size, spatial_indices, **kwargs):
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG

        batch_size = gt_boxes.shape[0]
        ret_dict = {
            'heatmaps': [],
            'target_boxes': [],
            'inds': [],
            'masks': [],
            'heatmap_masks': [],
            'target_boxes_src': [],
        }

        all_names = np.array(['bg', *self.class_names])
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            heatmap_list, target_boxes_list, inds_list, masks_list, target_boxes_src_list = [], [], [], [], []
            for bs_idx in range(batch_size):
                cur_gt_boxes = gt_boxes[bs_idx]
                gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]

                gt_boxes_single_head = []
                for idx, name in enumerate(gt_class_names):
                    if name not in cur_class_names:
                        continue
                    temp_box = cur_gt_boxes[idx]
                    temp_box[-1] = cur_class_names.index(name) + 1
                    gt_boxes_single_head.append(temp_box[None, :])

                if len(gt_boxes_single_head) == 0:
                    gt_boxes_single_head = cur_gt_boxes[:0, :]
                else:
                    gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)

                heatmap, ret_boxes, inds, mask, ret_boxes_src = self.assign_target_of_single_head(
                    num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head,
                    feature_map_size=feature_map_size, feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                    spatial_indices=spatial_indices[spatial_indices[:, 0] == bs_idx][:, [2, 1]],
                    num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
                    gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                    min_radius=target_assigner_cfg.MIN_RADIUS,
                )
                heatmap_list.append(heatmap.to(gt_boxes_single_head.device))
                target_boxes_list.append(ret_boxes.to(gt_boxes_single_head.device))
                inds_list.append(inds.to(gt_boxes_single_head.device))
                masks_list.append(mask.to(gt_boxes_single_head.device))
                target_boxes_src_list.append(ret_boxes_src.to(gt_boxes_single_head.device))

            ret_dict['heatmaps'].append(torch.cat(heatmap_list, dim=1).permute(1, 0))
            ret_dict['target_boxes'].append(target_boxes_list)
            ret_dict['inds'].append(inds_list)
            ret_dict['masks'].append(masks_list)
            ret_dict['target_boxes_src'].append(target_boxes_src_list)
        return ret_dict

    def get_loss(self):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']
        spatial_indices = self.forward_ret_dict['spatial_indices']

        loss, tb_dict = 0, {}
        for idx, pred_dict in enumerate(pred_dicts):

            # heatmap loss
            pred_dict['heatmap'] = torch.clamp(pred_dict['heatmap'].sigmoid(), min=1e-4, max=1 - 1e-4)
            hm_loss = loss_utils.focal_loss_sparse(pred_dict['heatmap'], target_dicts['heatmaps'][idx])
            hm_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']

            # reg loss
            target_boxes = target_dicts['target_boxes'][idx]
            pred_boxes = torch.cat([pred_dict[head_name] for head_name in self.separate_head_cfg.HEAD_ORDER], dim=1)
            reg_loss = loss_utils.reg_loss_sparse(
                pred_boxes, target_dicts['masks'][idx], target_dicts['inds'][idx], target_boxes, spatial_indices)
            loc_loss = (reg_loss * reg_loss.new_tensor(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'])).sum()
            loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']

            loss += hm_loss + loc_loss
            tb_dict['hm_loss_head_%d' % idx] = hm_loss.item()
            tb_dict['loc_loss_head_%d' % idx] = loc_loss.item()

            # iou loss
            if 'iou' in pred_dict:
                batch_box_preds = centernet_utils.decode_bbox_from_sparse_pred_dicts(
                    pred_dict, spatial_indices[:, 1:], self.point_cloud_range, self.voxel_size, self.feature_map_stride)
                iou_loss = loss_utils.iou_loss_sparse(
                    pred_dict['iou'], target_dicts['masks'][idx], target_dicts['inds'][idx],
                    batch_box_preds.clone().detach(), target_dicts['target_boxes_src'][idx], spatial_indices)

                loss += iou_loss
                tb_dict['iou_loss_head_%d' % idx] = iou_loss.item()

        tb_dict['rpn_loss'] = loss.item()
        return loss, tb_dict

    def generate_predicted_boxes(self, batch_size, pred_dicts, sparse_indices):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).cuda().float()

        ret_dict = [{'pred_boxes': [], 'pred_scores': [], 'pred_labels': []} for _ in range(batch_size)]
        for idx, pred_dict in enumerate(pred_dicts):
            batch_hm = pred_dict['heatmap'].sigmoid()
            batch_center = pred_dict['center']
            batch_center_z = pred_dict['center_z']
            batch_dim = pred_dict['dim'].exp()
            batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
            batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
            batch_vel = pred_dict['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None
            batch_iou = (pred_dict['iou'] + 1) * 0.5 if 'iou' in pred_dict else None

            final_pred_dicts = centernet_utils.decode_bbox_from_sparse_heatmap(
                heatmap=batch_hm, rot_cos=batch_rot_cos, rot_sin=batch_rot_sin,
                center=batch_center, center_z=batch_center_z, dim=batch_dim, vel=batch_vel, iou=batch_iou,
                point_cloud_range=self.point_cloud_range,
                voxel_size=self.voxel_size,
                feature_map_stride=self.feature_map_stride,
                sparse_indices=sparse_indices,
                K=post_process_cfg.MAX_OBJ_PER_SAMPLE,
                score_thresh=post_process_cfg.SCORE_THRESH,
                post_center_limit_range=post_center_limit_range
            )

            for bidx, final_dict in enumerate(final_pred_dicts):
                final_dict['pred_labels'] = self.class_id_mapping_each_head[idx][final_dict['pred_labels'].long()]

                if post_process_cfg.get('USE_IOU_TO_RECTIFY_SCORE', False) and 'pred_ious' in final_dict:
                    pred_scores, pred_labels, pred_ious = final_dict['pred_scores'], final_dict['pred_labels'], final_dict['pred_ious']
                    IOU_RECTIFIER = pred_scores.new_tensor(post_process_cfg.IOU_RECTIFIER)
                    final_dict['pred_scores'] = torch.pow(pred_scores, 1 - IOU_RECTIFIER[pred_labels]) * torch.pow(pred_ious, IOU_RECTIFIER[pred_labels])

                if post_process_cfg.NMS_CONFIG.NMS_TYPE == 'class_specific_nms':
                    selected, selected_scores = model_nms_utils.class_specific_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        box_labels=final_dict['pred_labels'], nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.NMS_CONFIG.get('SCORE_THRESH', None)
                    )
                else:
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        nms_config=post_process_cfg.NMS_CONFIG, score_thresh=None,
                    )

                ret_dict[bidx]['pred_boxes'].append(final_dict['pred_boxes'][selected])
                ret_dict[bidx]['pred_scores'].append(selected_scores)
                ret_dict[bidx]['pred_labels'].append(final_dict['pred_labels'][selected])

        for bidx in range(batch_size):
            ret_dict[bidx]['pred_boxes'] = torch.cat(ret_dict[bidx]['pred_boxes'], dim=0)
            ret_dict[bidx]['pred_scores'] = torch.cat(ret_dict[bidx]['pred_scores'], dim=0)
            ret_dict[bidx]['pred_labels'] = torch.cat(ret_dict[bidx]['pred_labels'], dim=0) + 1

        return ret_dict

    def forward(self, data_dict):
        x = data_dict['spatial_features_2d']
        spatial_indices = None

        pred_dicts = []
        for head in self.heads_list:
            pred_dict = head(x)
            for k, v in pred_dict.items():
                spatial_indices = v.indices
                pred_dict[k] = v.features
            pred_dicts.append(pred_dict)
        self.forward_ret_dict['pred_dicts'] = pred_dicts
        self.forward_ret_dict['spatial_indices'] = spatial_indices

        if self.training:
            target_dicts = self.assign_targets(data_dict['gt_boxes'], x.spatial_shape[-2:], spatial_indices)
            self.forward_ret_dict['target_dicts'] = target_dicts
        else:
            data_dict['final_box_dicts'] = self.generate_predicted_boxes(data_dict['batch_size'], pred_dicts, spatial_indices)

        return data_dict
