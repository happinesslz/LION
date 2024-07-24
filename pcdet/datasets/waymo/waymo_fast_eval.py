# OpenPCDet PyTorch Dataloader and Evaluation Tools for Waymo Open Dataset
# Reference https://github.com/open-mmlab/OpenPCDet
# Written by Shaoshuai Shi, Chaoxu Guo
# All Rights Reserved 2019-2020.
import multiprocessing
import os
from functools import partial

import numpy as np
import pickle
import tensorflow as tf
import tqdm
from os.path import join
import time
from waymo_open_dataset.protos import metrics_pb2
from waymo_open_dataset import label_pb2
import argparse

tf.get_logger().setLevel('INFO')


def limit_period(val, offset=0.5, period=np.pi):
    return val - np.floor(val / period + offset) * period


class OpenPCDetWaymoDetectionMetricsEstimator(tf.test.TestCase):
    WAYMO_CLASSES = ['unknown', 'Vehicle', 'Pedestrian', 'Truck', 'Cyclist']


    def generate_waymo_type_results(self, infos, class_names, is_gt=False, fake_gt_infos=True):
        def boxes3d_kitti_fakelidar_to_lidar(boxes3d_lidar):
            """
            Args:
                boxes3d_fakelidar: (N, 7) [x, y, z, w, l, h, r] in old LiDAR coordinates, z is bottom center

            Returns:
                boxes3d_lidar: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
            """
            w, l, h, r = boxes3d_lidar[:, 3:4], boxes3d_lidar[:, 4:5], boxes3d_lidar[:, 5:6], boxes3d_lidar[:, 6:7]
            boxes3d_lidar[:, 2] += h[:, 0] / 2
            return np.concatenate([boxes3d_lidar[:, 0:3], l, w, h, -(r + np.pi / 2)], axis=-1)

        frame_id, boxes3d, obj_type, score, overlap_nlz, difficulty = [], [], [], [], [], []
        context_name, timestamp_micros = list(), list()

        for frame_index, info in enumerate(infos):
            if is_gt:
                box_mask = np.array([n in class_names for n in info['name']], dtype=np.bool_)
                if 'num_points_in_gt' in info:
                    zero_difficulty_mask = info['difficulty'] == 0
                    info['difficulty'][(info['num_points_in_gt'] > 5) & zero_difficulty_mask] = 1
                    info['difficulty'][(info['num_points_in_gt'] <= 5) & zero_difficulty_mask] = 2
                    nonzero_mask = info['num_points_in_gt'] > 0
                    box_mask = box_mask & nonzero_mask
                else:
                    print('Please provide the num_points_in_gt for evaluating on Waymo Dataset '
                          '(If you create Waymo Infos before 20201126, please re-create the validation infos '
                          'with version 1.2 Waymo dataset to get this attribute). SSS of OpenPCDet')
                    raise NotImplementedError

                num_boxes = box_mask.sum()
                box_name = info['name'][box_mask]

                difficulty.append(info['difficulty'][box_mask])
                score.append(np.ones(num_boxes))
                if fake_gt_infos:
                    info['gt_boxes_lidar'] = boxes3d_kitti_fakelidar_to_lidar(info['gt_boxes_lidar'])

                if info['gt_boxes_lidar'].shape[-1] == 9:
                    boxes3d.append(info['gt_boxes_lidar'][box_mask][:, 0:7])
                else:
                    boxes3d.append(info['gt_boxes_lidar'][box_mask])
            else:
                num_boxes = len(info['boxes_lidar'])
                difficulty.append([0] * num_boxes)
                score.append(info['score'])
                boxes3d.append(np.array(info['boxes_lidar'][:, :7]))
                box_name = info['name']
                context_name.extend([info['metadata']['context_name']])
                timestamp_micros.extend([info['metadata']['timestamp_micros']])
                if boxes3d[-1].shape[-1] == 9:
                    boxes3d[-1] = boxes3d[-1][:, 0:7]

            obj_type += [self.WAYMO_CLASSES.index(name) for i, name in enumerate(box_name)]
            frame_id.append(np.array([frame_index] * num_boxes))
            overlap_nlz.append(np.zeros(num_boxes))  # set zero currently

        frame_id = np.concatenate(frame_id).reshape(-1).astype(np.int64)
        boxes3d = np.concatenate(boxes3d, axis=0)
        obj_type = np.array(obj_type).reshape(-1)
        score = np.concatenate(score).reshape(-1)
        overlap_nlz = np.concatenate(overlap_nlz).reshape(-1)
        difficulty = np.concatenate(difficulty).reshape(-1).astype(np.int8)

        boxes3d[:, -1] = limit_period(boxes3d[:, -1], offset=0.5, period=np.pi * 2)

        return frame_id, boxes3d, obj_type, score, overlap_nlz, difficulty, context_name, timestamp_micros

    def eval_value_ops(self, sess, graph, metrics):
        return {item[0]: sess.run([item[1][0]]) for item in metrics.items()}

    def mask_by_distance(self, distance_thresh, boxes_3d, *args):
        mask = np.linalg.norm(boxes_3d[:, 0:2], axis=1) < distance_thresh + 0.5
        boxes_3d = boxes_3d[mask]
        ret_ans = [boxes_3d]
        for arg in args:
            ret_ans.append(arg[mask])

        return tuple(ret_ans)

    def parse_objects_single(self, index, pd_frameid, pd_boxes3d, pd_type, pd_score, context_name, timestamp_micros):

        def parse_one_object(boxes3d, score, cls_type, context, frame_timestamp):
            box = label_pb2.Label.Box()
            box.center_x = boxes3d[0]
            box.center_y = boxes3d[1]
            box.center_z = boxes3d[2]
            box.length = boxes3d[3]
            box.width = boxes3d[4]
            box.height = boxes3d[5]
            box.heading = boxes3d[6]

            o = metrics_pb2.Object()
            o.object.box.CopyFrom(box)
            o.object.type = cls_type
            o.score = score

            o.context_name = context
            o.frame_timestamp_micros = frame_timestamp

            return o

        objects_list = list()

        mask = pd_frameid == index
        temp_pd_boxes3d = pd_boxes3d[mask]
        temp_pd_score = pd_score[mask]
        temp_pd_type = pd_type[mask]

        for j in range(len(temp_pd_boxes3d)):
            pd_o = parse_one_object(temp_pd_boxes3d[j], temp_pd_score[j], temp_pd_type[j], context_name[index],
                                 timestamp_micros[index])
            objects_list.append(pd_o)
        return objects_list

    def parse_objects(self, pd_frameid, pd_boxes3d, pd_type, pd_score, context_name, timestamp_micros, workers=64):

        objects = metrics_pb2.Objects()

        parse_objects_single = partial(
            self.parse_objects_single,
            pd_frameid=pd_frameid, pd_boxes3d=pd_boxes3d,
            pd_type=pd_type, pd_score=pd_score, context_name=context_name,
            timestamp_micros=timestamp_micros)

        with multiprocessing.Pool(workers) as p:
            all_list = list(p.map(parse_objects_single, np.arange(0, len(context_name))))

        progress_bar = tqdm.tqdm(total=len(all_list), leave=True, desc='convert', dynamic_ncols=True)
        for cur_list in all_list:
            for i in range(len(cur_list)):
                objects.objects.append(cur_list[i])
            progress_bar.update()
        progress_bar.close()

        return objects

    def waymo_evaluation(self, prediction_infos, gt_infos, class_name, pathname=None, distance_thresh=100,
                         fake_gt_infos=True, workers=64, root=None):
        print('Start the fast waymo evaluation...')

        if pathname is None:
            pathname = 'tmp'

        if root is None:
            root = '..'

        assert len(prediction_infos) == len(gt_infos), '%d vs %d' % (prediction_infos.__len__(), gt_infos.__len__())

        pred_path = join(root, 'fast_output', pathname, pathname + '.bin')
        gt_path = join(root, 'fast_output', 'gt.bin')

        existed = os.path.exists(pred_path)

        if not existed or pathname == 'tmp':
            pd_frameid, pd_boxes3d, pd_type, pd_score, pd_overlap_nlz, _, context_name, timestamp_micros = self.generate_waymo_type_results(
                prediction_infos, class_name, is_gt=False
            )

            pd_boxes3d, pd_frameid, pd_type, pd_score, pd_overlap_nlz = self.mask_by_distance(
                distance_thresh, pd_boxes3d, pd_frameid, pd_type, pd_score, pd_overlap_nlz
            )

            start_time = time.time()
            objects = self.parse_objects(pd_frameid, pd_boxes3d, pd_type, pd_score, context_name,
                                                     timestamp_micros, workers)

            if not os.path.exists(join(root, 'fast_output', pathname)):
                os.makedirs(join(root, 'fast_output', pathname))

            with open(pred_path, 'wb') as f:
                f.write(objects.SerializeToString())

            end_time = time.time()
            print(f'Convert End, Time: {(end_time - start_time)}s')

        import subprocess
        command = f'{root}/' + f'compute_detection_metrics_main {pred_path} {gt_path}' 
        # command = f'{root}/faster-waymo-detection-evaluation/bazel-bin/waymo_open_dataset/metrics/tools/' + f'compute_detection_metrics_main {pred_path} {gt_path}'

        ret_bytes = subprocess.check_output(command, shell=True)
        ret_texts = ret_bytes.decode('utf-8')

        # parse the text to get ap_dict
        ap_dict = {
            'Vehicle/L1 mAP': 0,
            'Vehicle/L1 mAPH': 0,
            'Vehicle/L2 mAP': 0,
            'Vehicle/L2 mAPH': 0,
            'Pedestrian/L1 mAP': 0,
            'Pedestrian/L1 mAPH': 0,
            'Pedestrian/L2 mAP': 0,
            'Pedestrian/L2 mAPH': 0,
            'Sign/L1 mAP': 0,
            'Sign/L1 mAPH': 0,
            'Sign/L2 mAP': 0,
            'Sign/L2 mAPH': 0,
            'Cyclist/L1 mAP': 0,
            'Cyclist/L1 mAPH': 0,
            'Cyclist/L2 mAP': 0,
            'Cyclist/L2 mAPH': 0,
            'Overall/L1 mAP': 0,
            'Overall/L1 mAPH': 0,
            'Overall/L2 mAP': 0,
            'Overall/L2 mAPH': 0
        }
        mAP_splits = ret_texts.split('mAP ')
        mAPH_splits = ret_texts.split('mAPH ')
        for idx, key in enumerate(ap_dict.keys()):
            split_idx = int(idx / 2) + 1
            if idx % 2 == 0:  # mAP
                ap_dict[key] = float(mAP_splits[split_idx].split(']')[0])
            else:  # mAPH
                ap_dict[key] = float(mAPH_splits[split_idx].split(']')[0])
        ap_dict['Overall/L1 mAP'] = \
            (ap_dict['Vehicle/L1 mAP'] + ap_dict['Pedestrian/L1 mAP'] +
             ap_dict['Cyclist/L1 mAP']) / 3
        ap_dict['Overall/L1 mAPH'] = \
            (ap_dict['Vehicle/L1 mAPH'] + ap_dict['Pedestrian/L1 mAPH'] +
             ap_dict['Cyclist/L1 mAPH']) / 3
        ap_dict['Overall/L2 mAP'] = \
            (ap_dict['Vehicle/L2 mAP'] + ap_dict['Pedestrian/L2 mAP'] +
             ap_dict['Cyclist/L2 mAP']) / 3
        ap_dict['Overall/L2 mAPH'] = \
            (ap_dict['Vehicle/L2 mAPH'] + ap_dict['Pedestrian/L2 mAPH'] +
             ap_dict['Cyclist/L2 mAPH']) / 3

        return ap_dict

def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--pred_infos', type=str, default=None, help='pickle file')
    parser.add_argument('--gt_infos', type=str, default=None, help='pickle file')
    parser.add_argument('--extra_tag', type=str, default=None, help='out path')
    parser.add_argument('--workers', type=int, default=64, help='workers')
    parser.add_argument('--class_names', type=str, nargs='+', default=['Vehicle', 'Pedestrian', 'Cyclist'], help='')
    parser.add_argument('--sampled_interval', type=int, default=5, help='sampled interval for GT sequences')
    args = parser.parse_args()

    assert args.extra_tag is not None, 'extra_tag is None'

    pred_infos = pickle.load(open(args.pred_infos, 'rb'))
    gt_infos = pickle.load(open(args.gt_infos, 'rb'))

    print('Start to evaluate the waymo format results...')
    eval = OpenPCDetWaymoDetectionMetricsEstimator()

    gt_infos_dst = []
    for idx in range(0, len(gt_infos), args.sampled_interval):
        cur_info = gt_infos[idx]['annos']
        cur_info['frame_id'] = gt_infos[idx]['frame_id']
        gt_infos_dst.append(cur_info)

    root = '.'

    ap_dict = eval.waymo_evaluation(
        pred_infos, gt_infos_dst, pathname=args.extra_tag, class_name=args.class_names, distance_thresh=1000,
        fake_gt_infos=False, workers=args.workers, root=root
    )

    print(ap_dict)


if __name__ == '__main__':
    main()
