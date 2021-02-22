# Get_started

## Data
Our work is based on Kitti Detection Dataset.
Your dataset folder should be organized like this:

```
PV_ENcoNet

├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2
│   │   │── kitti_dbinfos_train.pkl
│   │   │── kitti_infos_train.pkl
│   │   │── kitti_infos_trainval.pkl
│   │   │── kitti_infos_val.pkl
├── pcdet
├── tools

```
## Training & Testing
### Quick demo 

```
python demo.py --cfg_file ${CONFIG_FILE} \
    --ckpt ${CKPT}\
    --data_path ${POINT_CLOUD_DATA}
```
Here ${POINT_CLOUD_DATA_PATH} could be the following format:

```
├── data
│   ├── you custom dataset
│   │   │── point_cloud_data
│   │   │   |── 000000.npy
│   │   │   |── 000000.npy

```


Then you can get result like this. 

### Test and evaluate the pretrained models

Test with a pretrained model:


```
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
```

To test all the saved checkpoints of a specific training setting and draw the performance curve on the Tensorboard, add the --eval_all argument:
```
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --eval_all
```
### Train a model

You can change config in CONFIG_FILE.
```
python train.py --cfg_file ${CONFIG_FILE}
```
