import os
import shutil
import yaml
from pathlib import Path
from easydict import EasyDict
import random
from pcdet.datasets.kitti import kitti_tracking_dataset
seq_frame_count = {0: 154, 1: 447, 2: 233, 3: 144, 4: 314, 5: 297, 6: 270, 7: 800, 8: 390, 9: 803, \
                   10: 294, 11: 373, 12: 78, 13: 340, 14: 106, 15: 376, 16: 209, 17: 145, 18: 339, 19: 1059, 20: 837}

def get_sequece_frame(velodyne_path):
    ## cal each sequence's frame count ##
    frame_count = {}
    for seq in range(0,21):
        print(seq)
        velodyne_path_seq = os.path.join(velodyne_path,str(seq).zfill(4))
        #_,_,files = os.walk(velodyne_path_seq)
        count = 0
        for _,_,files in os.walk(velodyne_path_seq):
            for f in files:
                f = f.split('.')[-1]
                if f =='bin':
                    count+=1
        frame_count.update({seq:count})
    print(frame_count)

def copy_file(source,desti,type,seq_frame_count):
    # image velo can use this method
    index = 0
    if not os.path.exists(desti):
        os.mkdir(desti)
    for seq in range(0,21):
        print(seq)
        f_count = seq_frame_count[seq]
        source_seq = os.path.join(source,str(seq).zfill(4))
        for _,_,files in os.walk(source_seq):
            for f in files:
                f_type = f.split('.')[-1]
                if f_type == type:
                    f_index = int(f.split('.')[0])
                    s = os.path.join(source_seq,f)
                    d = desti + str(f_index+index).zfill(6) + '.' + type
                    if seq == 20:
                        print(f,f_index+index,d)
                        shutil.copyfile(s,d)
        index += f_count

def copy_file_label(source,desti,seq_frame_count):

    if not os.path.exists(desti):
        os.mkdir(desti)
    seq_idx = 0
    for seq in range(0, 21):
        source_seq = os.path.join(source,str(seq).zfill(4)+'.txt')
        f_count = seq_frame_count[seq]
        detection_dict = {} # frame_idx : [obj1_labels,obj2_labels,...]有效位数为2位 int:list
        with open(source_seq, "r") as f: #读取该序列的label, 将序列每一帧的标注结果存成一个label
            for line in f.readlines():
                line = line.strip('\n')  # 去掉列表中每一个元素的换行符
                line = line.split(' ')
                f_idx = int(line[0]) + seq_idx # 将每个序列的编号 更新为整体编号
                f_val = line[2:]
                if f_idx not in detection_dict:
                    f_val = [f_val]
                    detection_dict.update({f_idx:f_val})
                else:
                    ## 该帧已经不是第一次被存入数据；
                    det_tmp = detection_dict[f_idx]
                    #print(f_idx)
                    det_tmp.append(f_val)
                    detection_dict.update({f_idx: det_tmp})

        seq_idx += f_count
        ##防止dict过大，每个seq写入一次
        for key, value in detection_dict.items():
            if int(key) >=1820 and int(key) <= 1850:
                print(key)
            desti_f = os.path.join(desti, str(key).zfill(6) + '.txt')
            with open(desti_f, "w") as f:
                for v in value:
                    f.write(str(v[0]))
                    for i in range(1,len(v)):
                        f.write(' '+str('%.2f' % float(v[i])))
                    f.write('\n')
        print('seq:',seq,'over')
        print(seq_idx)

def copy_file_calib(source,desti,seq_frame_count):

    if not os.path.exists(desti):
        os.mkdir(desti)
    seq_idx = 0
    for seq in range(0, 21):
        f_count = seq_frame_count[seq]
        source_seq = os.path.join(source, str(seq).zfill(4)+'.txt')
        for idx in range(seq_idx,seq_idx+f_count):
            desti_f = os.path.join(desti,str(idx).zfill(6)+'.txt')
            shutil.copyfile(source_seq, desti_f)
        seq_idx += f_count
        print('seq:',seq,'over')

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
    frame_train = frame[:-50]
    frame_val = frame[-50:]
    with open(desti_f, "w") as f:  # 格式化字符串还能这么用！
        for f_idx in frame_train:
            f.write(str(f_idx).zfill(6))
            f.write('\n')
    desti_f = os.path.join(Root_dir, 'ImageSets', 'val.txt')
    with open(desti_f, "w") as f:  # 格式化字符串还能这么用！
        for f_idx in frame_val:
            f.write(str(f_idx).zfill(6))
            f.write('\n')

def check_label_continue(path):
    #由于有一些帧中不含有目标 会导致label的不连续
    l = 'DontCare -1.00 -1.00 -10.00 206.69 166.17 227.07 184.60 -1000.00 -1000.00 -1000.00 -10.00 -1.00 -1.00 -1.00'
    for i in range(8008):
        file = path + str(i).zfill(6) +'.txt'
        if not os.path.exists(file):
            with open(file,'w') as f:
                f.write(l)
            f.close()
            print('create lable idx:',file)
Root_dir = '/media/ddd/data2/kitti_tracking2detection/'
sourcevelodyne_path = '/media/ddd/data2/kitti_tracking/label_02/'
desti = '/media/ddd/data2/kitti_tracking2detection/label_2/'
sourcevelodyne_path = '/media/ddd/data2/KITTI_MOTS/train/images/'
desti = '/media/ddd/data2/kitti_tracking2detection/image_2/'
#(sourcevelodyne_path)
#create_val_idx(Root_dir,seq_frame_count)
copy_file(sourcevelodyne_path,desti,'png',seq_frame_count)
#copy_file_label(sourcevelodyne_path,desti,seq_frame_count)
#copy_file_calib(sourcevelodyne_path,desti,seq_frame_count)
## 创建训练集的idx
#create_training_idx(Root_dir,8008)
#check_label_continue(desti)
## 创建kitti detection 的训练集合/测试集合，建立数据增强所需要的gt 数据
dataset_cfg_path = '/media/ddd/data2/3d_MOTS_Ex./Code/OpenPCDet-RandlaNet/tools/cfgs/dataset_configs/kitti_dataset.yaml'
dataset_cfg = yaml.load(open(dataset_cfg_path))
dataset_cfg = EasyDict(dataset_cfg)
ROOT_DIR = Path('/media/ddd/data2/kitti_tracking2detection/')
kitti_tracking_dataset.create_kitti_infos(
        dataset_cfg=dataset_cfg,
        class_names=['Car', 'Pedestrian', 'Cyclist'],
        data_path=ROOT_DIR,
        save_path=Path(ROOT_DIR)
    )
for seq in range(7,7):
    print('seq', seq)
    val_seq = os.path.join(ROOT_DIR, 'ImageSets', str(seq).zfill(2))
    source_seq = os.path.join(val_seq, 'val.txt')
    desti_f = ROOT_DIR /'ImageSets' / 'val.txt'
    #shutil.copyfile(source_seq, desti_f)
    kitti_tracking_dataset.create_kitti_infos(
        dataset_cfg=dataset_cfg,
        class_names=['Car', 'Pedestrian', 'Cyclist'],
        data_path=ROOT_DIR,
        save_path=Path(ROOT_DIR)
    )
    '''
    kitti_tracking_dataset.create_kitti_infos(
        dataset_cfg=dataset_cfg,
        class_names=['Car', 'Pedestrian', 'Cyclist'],
        data_path=ROOT_DIR,
        save_path=Path(val_seq)
    )'''
