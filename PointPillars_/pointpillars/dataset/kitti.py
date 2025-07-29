import numpy as np
import os
import torch
from torch.utils.data import Dataset

import sys
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE))

from pointpillars.utils import read_pickle, read_points, bbox_camera2lidar
from pointpillars.dataset import point_range_filter, data_augment


class BaseSampler():
    """
    handles random sampling of GT objects from db
    """
    def __init__(self, sampled_list, shuffle=True):
        self.total_num = len(sampled_list)
        self.sampled_list = np.array(sampled_list)
        self.indices = np.arange(self.total_num)
        if shuffle:
            np.random.shuffle(self.indices)
        self.shuffle = shuffle
        self.idx = 0

    def sample(self, num):
        if self.idx + num < self.total_num:
            ret = self.sampled_list[self.indices[self.idx:self.idx+num]]
            self.idx += num
        else:
            ret = self.sampled_list[self.indices[self.idx:]]
            self.idx = 0
            if self.shuffle:
                np.random.shuffle(self.indices)
        return ret


class Kitti(Dataset):

    CLASSES = {
        'Pedestrian': 0, 
        'Cyclist': 1, 
        'Car': 2
        }

    def __init__(self, data_root, split, pts_prefix='velodyne_reduced'):
        assert split in ['train', 'val', 'trainval', 'test']
        self.data_root = data_root
        self.split = split
        self.pts_prefix = pts_prefix
        self.data_infos = read_pickle(os.path.join(data_root, f'kitti_infos_{split}.pkl')) # kitti_infos_train.pkl
        self.sorted_ids = list(self.data_infos.keys())
        db_infos = read_pickle(os.path.join(data_root, 'kitti_dbinfos_train.pkl'))
        db_infos = self.filter_db(db_infos)
        
        """
        data_infos = {
            000001: {
                'velodyne_path':
                'image':
                'calib':
                'annos':
            }
            ...
        }
        
        db_infos = {
            'Car': [db_info1, db_info2, ...],
            'Pedestrian': [...],
            ...
        }
        
        db_info = {
            'name':
            'path':
            'box3d_lidar':
            'difficulty': 
            'num_points_in_gt':
        }
        """

        db_sampler = {}
        # Initializes a BaseSampler for each object class (Car, Pedestrian, Cyclist)
        for cat_name in self.CLASSES:
            db_sampler[cat_name] = BaseSampler(db_infos[cat_name], shuffle=True)
        
        # Defines the data augmentation configuration
        self.data_aug_config=dict(
            # sampler
            db_sampler=dict(
                db_sampler=db_sampler,
                sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10)
                ),
            # variations
            object_noise=dict(
                num_try=100,
                translation_std=[0.25, 0.25, 0.25],
                rot_range=[-0.15707963267, 0.15707963267]
                ),
            # flip
            random_flip_ratio=0.5,
            # global transformation to the entire point cloud
            global_rot_scale_trans=dict(
                rot_range=[-0.78539816, 0.78539816],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0]
                ), 
            # Filters out points and objects outside the valid region
            point_range_filter=[-70.4, -40.0, -3, 70.4, 40.0, 1], ### FULL 360 RANGE !!!
            object_range_filter=[-70.4, -40.0, -3, 70.4, 40.0, 1]             
        )

    def remove_dont_care(self, annos_info):
        keep_ids = [i for i, name in enumerate(annos_info['name']) if name != 'DontCare']
        for k, v in annos_info.items():
            annos_info[k] = v[keep_ids]
        return annos_info

    def filter_db(self, db_infos):
        # 1. filter_by_difficulty
        for k, v in db_infos.items():
            db_infos[k] = [item for item in v if item['difficulty'] != -1]

        # 2. filter_by_min_points, dict(Car=5, Pedestrian=10, Cyclist=10)
        filter_thrs = dict(Car=5, Pedestrian=10, Cyclist=10)
        for cat in self.CLASSES:
            filter_thr = filter_thrs[cat]
            db_infos[cat] = [item for item in db_infos[cat] if item['num_points_in_gt'] >= filter_thr]
        
        return db_infos

    def __getitem__(self, index):
        data_info = self.data_infos[self.sorted_ids[index]]
        image_info, calib_info, annos_info = \
            data_info['image'], data_info['calib'], data_info['annos']
    
        # point cloud input
        velodyne_path = data_info['velodyne_path'].replace('velodyne', self.pts_prefix) # kitti/training/velodyne_reduced/000001.bin
        pts_path = os.path.join(self.data_root, velodyne_path)
        pts = read_points(pts_path) # (N, 4)
        
        # calib input: for bbox coordinates transformation between Camera and Lidar.
        tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32) # (4, 4)
        r0_rect = calib_info['R0_rect'].astype(np.float32) # (4, 4)

        # annotations input
        annos_info = self.remove_dont_care(annos_info)
        annos_name = annos_info['name']
        annos_location = annos_info['location']
        annos_dimension = annos_info['dimensions']
        rotation_y = annos_info['rotation_y']
        gt_bboxes = np.concatenate([annos_location, annos_dimension, rotation_y[:, None]], axis=1).astype(np.float32) # (num_valid_boxes, 7)
        # gt_bboxes_3d = bbox_camera2lidar(gt_bboxes, tr_velo_to_cam, r0_rect) # don't need since labels are in lidar frame
        gt_labels = [self.CLASSES.get(name, -1) for name in annos_name]
        
        data_dict = {
            'pts': pts, # (N, 4)
            'gt_bboxes_3d': gt_bboxes, # (num_valid_boxes, 7)
            'gt_labels': np.array(gt_labels), # (num_valid_boxes,), ex. [0, 1, 1]
            'gt_names': annos_name, # (num_valid_boxes,)
            'difficulty': annos_info['difficulty'],
            'image_info': image_info,
            'calib_info': calib_info
        }
        
        if self.split in ['train', 'trainval']:
            # everything should be in lidar frame
            data_dict = data_augment(self.CLASSES, self.data_root, data_dict, self.data_aug_config)
        else:
            # point_range_filter already adjusted in line 110
            data_dict = point_range_filter(data_dict, point_range=self.data_aug_config['point_range_filter'])

        return data_dict

    def __len__(self):
        return len(self.data_infos)
 

if __name__ == '__main__':
    
    kitti_data = Kitti(data_root='/mnt/ssd1/lifa_rdata/det/kitti', 
                       split='train')
    kitti_data.__getitem__(9)
