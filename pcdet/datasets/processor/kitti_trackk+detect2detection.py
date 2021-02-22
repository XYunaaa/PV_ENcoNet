import os
import shutil
import yaml
from pathlib import Path
from easydict import EasyDict
import random
from pcdet.datasets.kitti import kitti_tracking_dataset
seq_frame_count = {0: 154, 1: 447, 2: 233, 3: 144, 4: 314, 5: 297, 6: 270, 7: 800, 8: 390, 9: 803, \
                   10: 294, 11: 373, 12: 78, 13: 340, 14: 106, 15: 376, 16: 209, 17: 145, 18: 339, 19: 1059, 20: 837}

def get_train_frame(path):
    f = open(path,'r')
    train_list = f.readlines()
    train_list_result = []
    for t in train_list:
        train_list_result.append(int(t))
    return train_list_result

def get_velo_frame(velodyne_path):
    ## cal each sequence's frame count ##
    count = 0
    for _,_,files in os.walk(velodyne_path):
        for f in files:
            f = f.split('.')[-1]
            if f =='bin':
                count+=1
    return count

def copy_file(source1,source2,desti,type,train_list):

    # image velo can use this method
    index = 0
    if not os.path.exists(desti):
        os.makedirs(desti)
    print('Now we tracking :',source1)
    for _,_,files in os.walk(source1):
        files = sorted(files)
        for f in files:
            f_type = f.split('.')[-1]
            if f_type == type:
                f_index = int(f.split('.')[0])
                if f_index in train_list:
                    print(f_index)
                    s = os.path.join(source1,f)
                    d = desti + str(f_index).zfill(6) + '.' + type
                    os.symlink(s,d)
                    name = desti + str(index).zfill(6) + '.' + type
                    os.rename(d,name)
                    index += 1

    print('Now we tracking :', source2)
    for _,_,files in os.walk(source2):
        files = sorted(files)
        for f in files:
            f_type = f.split('.')[-1]
            if f_type == type:
                f_index = int(f.split('.')[0])
                print(f_index)
                s = os.path.join(source2,f)
                d = desti + str(f_index+index).zfill(6) + '.' +type
                os.symlink(s,d)

def create_val_idx(root_dir,seq_frame_count):
    if not os.path.exists(desti):
        os.mkdir(desti)
    seq_idx = 0
    for seq in range(0, 21):
        val_seq = os.path.join(root_dir, 'ImageSets', str(seq).zfill(2))
        if not os.path.exists(val_seq):
            os.mkdir(val_seq)
        val_seq = os.path.join(val_seq, 'val.txt')
        f_count = seq_frame_count[seq]

        val = open(val_seq,'w')
        for idx in range(seq_idx,seq_idx+f_count):
            idx = str(idx).zfill(6)
            val.write(idx+'\n')
        val.close()
        seq_idx += f_count
        print('seq:',seq,'over')

def create_training_idx(Root_dir,num):

    desti_f = os.path.join(Root_dir, 'ImageSets','train.txt')

    frame = [i for i in range(num)]
    random.shuffle(frame)
    frame_train = frame
    with open(desti_f, "w") as f:  # 格式化字符串还能这么用！
        for f_idx in frame_train:
            f.write(str(f_idx).zfill(6))
            f.write('\n')



train_list = get_train_frame('/media/ddd/data2/3d_MOTS_Ex./Code/OpenPCDet-Data/kitti_3000/ImageSets/train.txt')
ng_velo_s1 = '/media/ddd/data2/3d_MOTS_Ex./Code/OpenPCDet-Data/kitti_3000/training/Raw_velodyne/'
ng_velo_s2 = '/media/ddd/data2/kitti_tracking2detection/velodyne/'
ng_velo_d = '/media/ddd/data2/3d_MOTS_Ex./Code/OpenPCDet-Data/kitti3000+tracking/training/Raw_velodyne/'
copy_file(ng_velo_s1,ng_velo_s2,ng_velo_d,'bin',train_list)
velo_s1 = '/media/ddd/data2/3d_MOTS_Ex./Code/OpenPCDet-Data/kitti_3000/training/velodyne_m_vfe/'
velo_s2 = '/media/ddd/data2/3d_MOTS_Ex./Code/OpenPCDet-Data/kitti_tracking/training/velodyne_m_vfe/'
velo_d = '/media/ddd/data2/3d_MOTS_Ex./Code/OpenPCDet-Data/kitti3000+tracking/training/velodyne_m_vfe/'
#copy_file(velo_s1,velo_s2,velo_d,'npy',train_list)
img_s1 = '/media/ddd/data2/3d_MOTS_Ex./Code/OpenPCDet-Data/kitti_3000/training/image_2/'
img_s2 = '/media/ddd/data2/3d_MOTS_Ex./Code/OpenPCDet-Data/kitti_tracking/training/image_2/'
img_d = '/media/ddd/data2/3d_MOTS_Ex./Code/OpenPCDet-Data/kitti3000+tracking/training/image_2/'
#copy_file(img_s1,img_s2,img_d,'png',train_list)
calib_s1 = '/media/ddd/data2/3d_MOTS_Ex./Code/OpenPCDet-Data/kitti_3000/training/calib/'
calib_s2 = '/media/ddd/data2/3d_MOTS_Ex./Code/OpenPCDet-Data/kitti_tracking/training/calib/'
calib_d = '/media/ddd/data2/3d_MOTS_Ex./Code/OpenPCDet-Data/kitti3000+tracking/training/calib/'
#copy_file(calib_s1,calib_s2,calib_d,'txt',train_list)
label_s1 = '/media/ddd/data2/3d_MOTS_Ex./Code/OpenPCDet-Data/kitti_3000/training/label_2/'
label_s2 = '/media/ddd/data2/3d_MOTS_Ex./Code/OpenPCDet-Data/kitti_tracking/training/label_2/'
label_d = '/media/ddd/data2/3d_MOTS_Ex./Code/OpenPCDet-Data/kitti3000+tracking/training/label_2/'
#copy_file(label_s1,label_s2,label_d,'txt',train_list)

# kitti tracking+det3000  to detection-11719
Root_dir = '/media/ddd/data2/3d_MOTS_Ex./Code/OpenPCDet-Data/kitti3000+tracking/training/'
save_dir = '/media/ddd/data2/3d_MOTS_Ex./Code/OpenPCDet-Data/kitti3000+tracking/'
#create_training_idx(Root_dir,11720)
## 创建kitti detection 的训练集合/测试集合，建立数据增强所需要的gt 数据
dataset_cfg_path = '/media/ddd/data2/3d_MOTS_Ex./Code/OpenPCDet-RandlaNet/tools/cfgs/dataset_configs/kitti_dataset.yaml'
dataset_cfg = yaml.load(open(dataset_cfg_path))
dataset_cfg = EasyDict(dataset_cfg)
ROOT_DIR = Path(Root_dir)
'''
kitti_tracking_dataset.create_kitti_infos(
        dataset_cfg=dataset_cfg,
        class_names=['Car', 'Pedestrian', 'Cyclist'],
        data_path=ROOT_DIR,
        save_path=Path(save_dir)
    )
'''