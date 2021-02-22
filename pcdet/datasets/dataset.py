from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2
import torch.utils.data as torch_data

import nearest_neighbors
from ..utils import common_utils
from ..utils import calibration_kitti
from .augmentor.data_augmentor import DataAugmentor
from .processor.data_processor import DataProcessor
from .processor.point_feature_encoder import PointFeatureEncoder


def painted_point_cloud(calib_result, img, point_cloud):
    pts_img = calib_result[0]  # [N,3] in lidar to [N,2] in img
    pts_rect_depth = calib_result[1]
    pts_img = np.round(pts_img).astype(int)  # 四舍五入
    # after lidar to img,filtering points in img's range ;
    img_shape = img.shape
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
    pts_non_valid_flag = ~pts_valid_flag

    ## in img's range ; all lidar's nonground points##
    pts_img = pts_img[pts_valid_flag]
    pts_img[:, [0, 1]] = pts_img[:, [1, 0]]  # height,width to width height[1242,375] to [375,1242]
    row = pts_img[:, 0]
    col = pts_img[:, 1]
    img = img.astype(int)
    pointcloud_color = img[row, col, :]  # [N,3] l a b

    non_valid_point_cloud = point_cloud[pts_non_valid_flag]
    non_valid_point_cloud_color = np.zeros((non_valid_point_cloud.shape[0], 3))
    painted_point_cloud_non_b = np.hstack(
        (non_valid_point_cloud, non_valid_point_cloud_color))  # [N,6] x,y,z,r,g,b
    painted_point_cloud_b = np.hstack(
        (point_cloud[pts_valid_flag], pointcloud_color))  # [N,6] x,y,z,r,g,b
    painted_point_cloud_b = np.vstack((painted_point_cloud_b, painted_point_cloud_non_b))
    return painted_point_cloud_b

def sample_points(points, coords, num_points):

    if num_points < len(points):
        ## 如果选点数量小于 点的总数 ##
        pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
        pts_near_flag = pts_depth < 40.0
        ## 在40,米以外的点
        far_idxs_choice = np.where(pts_near_flag == 0)[0]
        ## 在40,米以内的点
        near_idxs = np.where(pts_near_flag == 1)[0]
        ## 在40米以内的点中随机选择num_points - len(far_idxs_choice)个
        near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
        choice = []
        # 若选点数大于40米以外的点总数
        if num_points > len(far_idxs_choice):
            ## 最终的选点为 40m以内的num_points - len(far_idxs_choice)个点 + 40米以外的所有点
            near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
            choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                if len(far_idxs_choice) > 0 else near_idxs_choice
        else:
            ##在所有点中 随机选择
            choice = np.arange(0, len(points), dtype=np.int32)
            choice = np.random.choice(choice, num_points, replace=False)
        np.random.shuffle(choice)
    else:
        choice = np.arange(0, len(points), dtype=np.int32)
        if num_points > len(points):
            if num_points - len(points) < len(points):
                extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
            else:
                extra_choice = np.random.choice(choice, num_points - len(points), replace=True)
            choice = np.concatenate((choice, extra_choice), axis=0)
        np.random.shuffle(choice)

    result = points[choice]
    coords_result = coords[choice]
    return result, coords_result

class DatasetTemplate(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.class_names = class_names
        self.logger = logger
        self.root_path = root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH)
        self.logger = logger
        if self.dataset_cfg is None or class_names is None:
            return

        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )
        self.data_augmentor = DataAugmentor(
            self.root_path, self.dataset_cfg.DATA_AUGMENTOR, self.class_names, logger=self.logger
        ) if self.training else None
        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range, training=self.training
        )

        self.grid_size = self.data_processor.grid_size
        self.voxel_size = self.data_processor.voxel_size
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False

    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        To support a custom dataset, implement this function to receive the predicted results from the model, and then
        transform the unified normative coordinate to your required coordinate, and optionally save them to disk.

        Args:
            batch_dict: dict of original data from the dataloader
            pred_dicts: dict of predicted results from the model
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path: if it is not None, save the results to this path
        Returns:

        """

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate and call the function self.prepare_data() to process the data and send them
        to the model.

        Args:
            index:

        Returns:

        """
        raise NotImplementedError

    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if self.training:
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)

            data_dict = self.data_augmentor.forward(
                data_dict={
                    **data_dict,
                    'gt_boxes_mask': gt_boxes_mask
                }
            )
            if len(data_dict['gt_boxes']) == 0:
                new_index = np.random.randint(self.__len__())
                return self.__getitem__(new_index)

        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes

        data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )
        data_dict.pop('gt_names', None)

        return data_dict
    '''
    @staticmethod
    def collate_batch(batch_list, _unused=False):

        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points','image']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points','raw_points','voxel_coords']:
                    coors = []
                    num = []
                    for i, coor in enumerate(val):
                        num.append(coor.shape[0])
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                    ## sample raw_points & voxels_feature
                    num_ = np.max(num)
                    coors_batch = []
                    for i, coor in enumerate(val):
                        p,_ = sample_points(coor,coor,num_)
                        coors_batch.append(p)
                    k = key+'_batch'
                    ret[k] = np.array(coors_batch).reshape((len(num),num_,-1))
                    ## knn for points after sampling
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                elif key in ['voxel_features']:
                    continue
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        val = data_dict['voxel_features']
        voxel_coords = data_dict['voxel_coords']
        num = []
        for i, coor in enumerate(val):
            num.append(coor.shape[0])
        num_ = np.max(num)
        fea = []
        cor = []
        for i, coor in enumerate(val):
            p, c = sample_points(coor, voxel_coords[i], num_)
            fea.append(p)
            cor.append(c)

        ret['voxel_features_batch'] = np.array(fea).reshape((len(num), num_, -1))
        ret['voxel_coords_batch'] = np.array(cor).reshape((len(num), num_, -1))
        neighbor_idx = nearest_neighbors.knn_batch(ret['raw_points_batch'], ret['voxel_features_batch'], 8, omp=True)
        ret['neighbor'] = neighbor_idx
        ret['batch_size'] = batch_size
        return ret
    '''

    def collate_batch(self, batch_list, _unused=False):

        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}
        image_idx = data_dict['frame_id']
        calib = data_dict['calib']
        for key, val in data_dict.items():
            try:
                if key in ['image']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points']:
                    num = []
                    for i, coor in enumerate(val):
                        num.append(coor.shape[0])
                    ## sample raw_points & voxels_feature
                    num_ = np.max(num)
                    coors_batch = []
                    #if num_ > 10000:
                    #    num_ = 10000
                    for i, coor in enumerate(val):
                        p, _ = sample_points(coor, coor, num_)
                        coors_batch.append(p)
                    k = key + '_batch'
                    ret[k] = np.array(coors_batch).reshape((len(num), num_, -1))
                elif key in ['raw_points']:
                    if self.raw == False: # dont need load raw data
                        #print(self.nbg)
                        num = []
                        for i, coor in enumerate(val):
                            num.append(coor.shape[0])
                        ## sample raw_points
                        num_ = np.max(num)
                        coors_batch = []
                        #if num_ > 10000:
                        #    num_ = 10000
                        for i, coor in enumerate(val):
                            p, _ = sample_points(coor, coor, num_)
                            if self.use_color:
                                img_path = '/media/ddd/data2/3d_MOTS_Ex./Code/OpenPCDet-RandlaNet/data/kitti/training/image_2/' + \
                                           image_idx[i] + '.png'
                                img = cv2.imread(img_path)
                                image = np.float32(img)
                                if not self.use_rgb:
                                    img = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
                                p = p[:,:3]
                                calib_result = calib[i].lidar_to_img(p)  # [N,3] in lidar to [N,2] in img
                                p = self.painted_point_cloud(calib_result,img,p)
                            coors_batch.append(p)
                        k = key + '_batch'
                        ret[k] = np.array(coors_batch).reshape((len(num), num_, -1)) # if colored,return have colored point cloud
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                elif key in ['voxel_features','voxels', 'voxel_num_points','voxel_coords']:
                    continue
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('image_idx:',image_idx)
                img_path = '/media/ddd/data2/3d_MOTS_Ex./Code/OpenPCDet-RandlaNet/data/kitti/training/image_2/' + \
                           image_idx[0] + '.png'
                img = cv2.imread(img_path)
                print(img_path)
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        val = data_dict['voxel_features']
        voxel_coords = data_dict['voxel_coords']
        num = []
        for i, coor in enumerate(val):
            num.append(coor.shape[0])
        num_ = np.max(num)
        fea = []
        cor = []
        #if num_ > 10000:
        #   num_ = 10000
        for i, coor in enumerate(val):
            p, c = sample_points(coor, voxel_coords[i], num_)
            if self.use_color:
                img_path = '/media/ddd/data2/3d_MOTS_Ex./Code/OpenPCDet-RandlaNet/data/kitti/training/image_2/' + \
                           image_idx[i] + '.png'
                img = cv2.imread(img_path)
                image = np.float32(img)
                if not self.use_rgb:
                    img = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
                p = p[:, :3]
                calib_result = calib[i].lidar_to_img(p)  # [N,3] in lidar to [N,2] in img
                p = self.painted_point_cloud(calib_result, img, p)
            fea.append(p)
            cor.append(c)

        ret['voxel_features_batch'] = np.array(fea).reshape((len(num), num_, -1)) #if colored,return have colored point cloud
        ret['voxel_coords_batch'] = np.array(cor).reshape((len(num), num_, -1))
        val = ret['voxel_coords_batch']
        coors = []
        for i, coor in enumerate(val):
            coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
            coors.append(coor_pad)
        ret['voxel_coords_batch'] = np.concatenate(coors, axis=0)
        voxel_features_batch = ret['voxel_features_batch'][:,:,:3]


        if self.nbg:# True ,使用有背景点的选点方式
            raw_points_batch = ret['raw_points_batch'][:, :, :3]
            neighbor_idx = nearest_neighbors.knn_batch(raw_points_batch, voxel_features_batch, 8, omp=True)
        else:
            neighbor_idx = nearest_neighbors.knn_batch(voxel_features_batch, voxel_features_batch, 8, omp=True)

        ret['neighbor'] = neighbor_idx
        ret['batch_size'] = batch_size

        return ret
