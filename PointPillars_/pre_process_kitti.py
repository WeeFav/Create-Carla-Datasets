import argparse
import pdb
import cv2
import numpy as np
import os
from tqdm import tqdm
import sys
CUR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CUR)

from pointpillars.utils import read_points, write_points, read_calib, read_label, \
    write_pickle, remove_outside_points, get_points_num_in_bbox, \
    points_in_bboxes_v2



def create_data_info_pkl(data_root, data_type, prefix, label=True, db=False):
    sep = os.path.sep
    print(f"Processing {data_type} data..")
    ids_file = os.path.join(CUR, 'pointpillars', 'dataset', 'ImageSets', f'{data_type}.txt')
    with open(ids_file, 'r') as f:
        ids = [id.strip() for id in f.readlines()]
    
    split = 'training' if label else 'testing'

    kitti_infos_dict = {} # will be saved as kitti_infos_train.pkl
    if db:
        kitti_dbinfos_train = {} # will be saved as kitti_dbinfos_train.pkl
        db_points_saved_path = os.path.join(data_root, f'{prefix}_gt_database') # /kitti_gt_database
        os.makedirs(db_points_saved_path, exist_ok=True)
        
    for id in tqdm(ids):
        cur_info_dict={}
        
        # paths
        img_path = os.path.join(data_root, split, 'image_2', f'{id}.png')
        lidar_path = os.path.join(data_root, split, 'velodyne', f'{id}.bin')
        calib_path = os.path.join(data_root, split, 'calib', f'{id}.txt') 
        cur_info_dict['velodyne_path'] = sep.join(lidar_path.split(sep)[-3:]) # -> training/velodyne/000001.bin

        # image info
        img = cv2.imread(img_path)
        image_shape = img.shape[:2]
        cur_info_dict['image'] = {
            'image_shape': image_shape,
            'image_path': sep.join(img_path.split(sep)[-3:]), # -> training/image_2/000001.png
            'image_idx': int(id),
        }

        # calib
        calib_dict = read_calib(calib_path)
        cur_info_dict['calib'] = calib_dict

        # lidar (reduced is same as original)
        lidar_points = read_points(lidar_path) # (N, 4)
        reduced_lidar_points = lidar_points # no need to remove lidar points
        saved_reduced_path = os.path.join(data_root, split, 'velodyne_reduced') # kitti/training/velodyne_reduced/
        os.makedirs(saved_reduced_path, exist_ok=True)
        saved_reduced_points_name = os.path.join(saved_reduced_path, f'{id}.bin') # kitti/training/velodyne_reduced/000001.bin
        write_points(reduced_lidar_points, saved_reduced_points_name)

        # label
        if label:
            label_path = os.path.join(data_root, split, 'label_2', f'{id}.txt')
            annotation_dict = read_label(label_path)
            annotation_dict['difficulty'] = np.zeros(annotation_dict['name'].shape[0]) # all bounding box have difficulty 0
            annotation_dict['num_points_in_gt'] = get_points_num_in_bbox( # ex. [15, -1, 7, 3], where each entry represents how many LiDAR points fall inside the corresponding 3D bounding box
                points=reduced_lidar_points,
                r0_rect=calib_dict['R0_rect'], 
                tr_velo_to_cam=calib_dict['Tr_velo_to_cam'],
                dimensions=annotation_dict['dimensions'],
                location=annotation_dict['location'],
                rotation_y=annotation_dict['rotation_y'],
                name=annotation_dict['name'])
            cur_info_dict['annos'] = annotation_dict

            if db:
                indices, n_total_bbox, n_valid_bbox, bboxes_lidar, name = \
                    points_in_bboxes_v2(
                        points=lidar_points,
                        r0_rect=calib_dict['R0_rect'].astype(np.float32), 
                        tr_velo_to_cam=calib_dict['Tr_velo_to_cam'].astype(np.float32),
                        dimensions=annotation_dict['dimensions'].astype(np.float32),
                        location=annotation_dict['location'].astype(np.float32),
                        rotation_y=annotation_dict['rotation_y'].astype(np.float32),
                        name=annotation_dict['name']    
                    )
                for j in range(n_valid_bbox):
                    # Centers the object point cloud at the origin so it can be reused and repositioned later.
                    db_points = lidar_points[indices[:, j]] # filter for points inside box j (n, 4)
                    db_points[:, :3] -= bboxes_lidar[j, :3] # Translate to local object coords
                    db_points_saved_name = os.path.join(db_points_saved_path, f'{int(id)}_{name[j]}_{j}.bin')
                    write_points(db_points, db_points_saved_name)

                    db_info={
                        'name': name[j],
                        'path': os.path.join(os.path.basename(db_points_saved_path), f'{int(id)}_{name[j]}_{j}.bin'),
                        'box3d_lidar': bboxes_lidar[j],
                        'difficulty': annotation_dict['difficulty'][j], 
                        'num_points_in_gt': len(db_points), 
                    }
                    if name[j] not in kitti_dbinfos_train:
                        kitti_dbinfos_train[name[j]] = [db_info]
                    else:
                        kitti_dbinfos_train[name[j]].append(db_info)
        
        kitti_infos_dict[int(id)] = cur_info_dict

    saved_path = os.path.join(data_root, f'{prefix}_infos_{data_type}.pkl')
    write_pickle(kitti_infos_dict, saved_path)
    if db:
        saved_db_path = os.path.join(data_root, f'{prefix}_dbinfos_train.pkl')
        write_pickle(kitti_dbinfos_train, saved_db_path)
    return kitti_infos_dict


def main(args):
    data_root = args.data_root
    prefix = args.prefix

    ## 1. train: create data infomation pkl file && create reduced point clouds 
    ##           && create database(points in gt bbox) for data aumentation
    kitti_train_infos_dict = create_data_info_pkl(data_root, 'train', prefix, db=True)

    ## 2. val: create data infomation pkl file && create reduced point clouds
    kitti_val_infos_dict = create_data_info_pkl(data_root, 'val', prefix)
    
    ## 3. trainval: create data infomation pkl file
    kitti_trainval_infos_dict = {**kitti_train_infos_dict, **kitti_val_infos_dict}
    saved_path = os.path.join(data_root, f'{prefix}_infos_trainval.pkl')
    write_pickle(kitti_trainval_infos_dict, saved_path)

    ## 4. test: create data infomation pkl file && create reduced point clouds
    kitti_test_infos_dict = create_data_info_pkl(data_root, 'test', prefix, label=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset infomation')
    parser.add_argument('--data_root', default='/mnt/ssd1/lifa_rdata/det/kitti', 
                        help='your data root for kitti')
    parser.add_argument('--prefix', default='kitti', 
                        help='the prefix name for the saved .pkl file')
    args = parser.parse_args()

    main(args)