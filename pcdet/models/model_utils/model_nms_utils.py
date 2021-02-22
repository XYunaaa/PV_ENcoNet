import torch
import os
import numpy as np
import time
from ...ops.iou3d_nms import iou3d_nms_utils


def class_agnostic_nms(box_scores, box_preds, nms_config, score_thresh=None,src_box_scores=None):
    start = time.time()
    src_box_scores = box_scores
     
    if score_thresh is not None:

        scores_mask = (box_scores >= score_thresh)
        
        path = '/media/ddd/data2/3d_MOTS_Ex./Code/OpenPCDet-RandlaNet/tools/time/'
        ''' 
        if os.path.exists(path+'post-boxshape1.npy'):
        	t = np.load(path+'post-boxshape1.npy')
        	t = list(t)
        	t.append(box_scores.shape[0])
        	np.save(path+'post-boxshape1.npy',t)
        else:
        	np.save(path+'post-boxshape1.npy',[box_scores.shape[0]])
        if os.path.exists(path+'post-score.npy'):
        	t = np.load(path+'post-score.npy')
        	t = list(t)
        	t.append(score_thresh)
        	np.save(path+'post-score.npy',t)
        else:
        	np.save(path+'post-score.npy',[score_thresh])
        '''     
        t1 = time.time()
        box_preds = box_preds[scores_mask]
        t2 = time.time()
        box_scores = box_scores[scores_mask]
        t3 = time.time()
    #print('	nms-part0:',t1-start)
        '''        
        if os.path.exists(path+'postprocessing-data.npy'):
        	t = np.load(path+'postprocessing-data.npy')
        	t = list(t)
        	t.append(t2-t1)
        	np.save(path+'postprocessing-data.npy',t)
        else:
        	np.save(path+'postprocessing-data.npy',[t2-t1])
              
        if os.path.exists(path+'post-boxshape2.npy'):
        	t = np.load(path+'post-boxshape2.npy')
        	t = list(t)
        	t.append(box_scores.shape[0])
        	np.save(path+'post-boxshape2.npy',t)
        else:
        	np.save(path+'post-boxshape2.npy',[(box_scores.shape[0])])
        '''         
    #print('	nms-part1:',t2-t1)
    #print('	nms-part2:',t3-t2)
    selected = []
    if box_scores.shape[0] > 0:
        t0 = time.time()
        box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
        topk = time.time()
        #print('	nms-topk:',topk-t0)
        boxes_for_nms = box_preds[indices]
        keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
                boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH, **nms_config
        )
        selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]
        #print('	nms-iou3d_nms_utils:',time.time()-topk)
    t1 = time.time()
    if score_thresh is not None:
        original_idxs = scores_mask.nonzero().view(-1)
        selected = original_idxs[selected]
    #print('	nms-score_thresh is not None:',time.time()-t1)
    return selected, src_box_scores[selected]


def multi_classes_nms(cls_scores, box_preds, nms_config, score_thresh=None):
    """
    Args:
        cls_scores: (N, num_class)
        box_preds: (N, 7 + C)
        nms_config:
        score_thresh:

    Returns:

    """
    pred_scores, pred_labels, pred_boxes = [], [], []
    for k in range(cls_scores.shape[1]):
        if score_thresh is not None:
            scores_mask = (cls_scores[:, k] >= score_thresh)
            box_scores = cls_scores[scores_mask, k]
            cur_box_preds = box_preds[scores_mask]
        else:
            box_scores = cls_scores[:, k]

        selected = []
        if box_scores.shape[0] > 0:
            box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
            boxes_for_nms = cur_box_preds[indices]
            keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
                    boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH, **nms_config
            )
            selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]

        pred_scores.append(box_scores[selected])
        pred_labels.append(box_scores.new_ones(len(selected)).long() * k)
        pred_boxes.append(cur_box_preds[selected])

    pred_scores = torch.cat(pred_scores, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    pred_boxes = torch.cat(pred_boxes, dim=0)

    return pred_scores, pred_labels, pred_boxes
