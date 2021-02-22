import torch.nn as nn
import time
import os
import numpy as np
class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        s = time.time()
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        #print('HightCompression : ',time.time()-s)
        t0 = time.time()
        '''         
        path = '/media/ddd/data2/3d_MOTS_Ex./Code/OpenPCDet-RandlaNet/tools/time/'
        if os.path.exists(path+'HightCompression.npy'):
        	t = np.load(path+'HightCompression.npy')
        	t = list(t)
        	t.append(t0-s)
        	np.save(path+'HightCompression.npy',t)
        else:
        	np.save(path+'HightCompression.npy',[t0-s])
        '''        
        return batch_dict
