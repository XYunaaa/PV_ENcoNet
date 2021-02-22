import argparse
import glob
import os
from pathlib import Path
import numpy as np
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
import pickle as pickle

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--use_color', type=bool, default=False, help='whether use colored pointed cloud')
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=True , help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--eval_distance', type=bool, default=False, help='eval by setting distance')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

def main():

    args, cfg = parse_config()
    output_dir = '/media/ddd/data2/3d_MOTS_Ex./Code/OpenPCDet-RandlaNet/tools/'

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False, workers=args.workers, training=False,use_color=cfg.USE_COLOR,
        use_rgb = cfg.USE_RGB,
        nbg = cfg.USE_NBG
    )
    result_path = '/media/ddd/data2/3d_MOTS_Ex./OpenPCDet-RandlaNet/output/kitti_models/Data/PV_ENcoNet_Lab/kitti3000+tracking/Final_result/epoch_54/val/result.pkl'
    det_annos = pickle.load(open(result_path, 'rb'))
    result_str, result_dict = test_loader.dataset.evaluation(
        det_annos, test_loader.dataset.class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=output_dir
    )
    print(result_str)


if __name__ == '__main__':
    main()
