import argparse
import glob
from pathlib import Path

from skimage import io
import mayavi.mlab as mlab
import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from visual_utils import visualize_utils as V
from pcdet.utils import calibration_kitti

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None,data_root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.data_root_path = data_root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path/ f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        data_file_list.sort()
        self.sample_file_list = data_file_list
        self.raw = False
        self.use_color = True
        self.nbg = True
        self.use_rgb = False

    def painted_point_cloud(self, calib_result, img, point_cloud):
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

    def get_lidar(self, idx):
        ## velodyne is filter_ground & painted's kitti lidar points ##
        lidar_file = self.data_root_path / 'velodyne' / ('%s.npy' % str(idx).zfill(6))

        assert lidar_file.exists()
        points = np.load(str(lidar_file))
        points = points[:, 0:3]
        return points

    def get_lidar_raw(self, idx):
        lidar_file = self.data_root_path / 'velodyne_m_vfe' / ('%s.npy' % str(idx).zfill(6))
        if not lidar_file.exists():
            print(lidar_file)
        assert lidar_file.exists()
        points_fov = np.load(str(lidar_file))
        return points_fov

    def get_calib(self, idx):
        calib_file = self.data_root_path / 'calib' / ('%s.txt' % str(idx).zfill(6))
        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)

    def get_image_shape(self, idx):
        img_file = self.data_root_path / 'image_2' / ('%s.png' % str(idx).zfill(6))
        assert img_file.exists()
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_fov_flag(self,pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        '''
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError
        '''
        print(index)
        points = self.get_lidar(index)

        raw_points_fov = self.get_lidar_raw(index)

        calib = self.get_calib(index)
        # kdtree = self.get_kdtree(sample_idx)
        img_shape = self.get_image_shape(index)

        pts_rect = calib.lidar_to_rect(points[:, 0:3])
        fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
        points = points[fov_flag]

        pts_rect = calib.lidar_to_rect(raw_points_fov[:, 0:3])
        fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
        raw_points_fov = raw_points_fov[fov_flag]

        input_dict = {
            'points': points,
            'frame_id': str(index).zfill(6),
            'calib': calib,
            'raw_points': raw_points_fov
        }
        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--root_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.npy', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.root_path),data_root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            print(f'Visualized sample index: \t{idx + 1}')
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            V.draw_scenes(
                points=data_dict['points_batch'][0, :, 0:], ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )
            mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
