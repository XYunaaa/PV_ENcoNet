import time
import os
import numpy as np
import torch
import torch.nn as nn
import spconv
from functools import partial
from ....utils import ciede2000

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m

class SharedMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        transpose=False,
        padding_mode='zeros',
        bn=False,
        activation_fn=None
    ):
        super(SharedMLP, self).__init__()

        conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d

        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding_mode=padding_mode
        )
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
        self.activation_fn = activation_fn

    def forward(self, input):
        r"""
            Forward pass of the network

            Parameters
            ----------
            input: torch.Tensor, shape (B, d_in, N, K)

            Returns
            -------
            torch.Tensor, shape (B, d_out, N, K)
        """
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x

class LocalSpatialEncoding(nn.Module):

    def __init__(self, d, num_neighbors, device,use_rgb=False):
        super(LocalSpatialEncoding, self).__init__()

        self.num_neighbors = num_neighbors
        self.mlp0 = SharedMLP(20, 16, bn=True, activation_fn=nn.ReLU())
        self.mlp1 = SharedMLP(16, d, bn=True, activation_fn=nn.ReLU())
        self.device = device
        self.use_rgb = use_rgb

    def forward(self, coords,raw_points, features, neigh_idx,whether_cal=1,ste_feature=None):
        r"""
            Forward pass

            Parameters
            ----------
            coords: torch.Tensor, shape (B, N, 6)
                coordinates of the non-ground point cloud
            raw_points: torch.Tensor, shape (B, M, 6)
                coordinates of the all point cloud
            features: torch.Tensor, shape (B, d, N, 1)
                features of the point cloud
            neighbors: tuple

            Returns
            -------
            torch.Tensor, shape (B, 2*d, N, K)
        """
        # finding neighboring points
        idx = neigh_idx.long()
        B, N, K = idx.size()
        if whether_cal==1:
            idx = neigh_idx.long()
            B, N, K = idx.size()
            M = raw_points.shape[1]

            extended_coords_xyzlab = coords.transpose(-2, -1).unsqueeze(-1).expand(B, 6, N, K) # CONTAINS XYZLAB
            extended_idx = idx.unsqueeze(1).expand(B, 6, N, K)
            extended_rawpoints_xyzlab = raw_points.transpose(-2, -1).unsqueeze(-1).expand(B, 6, M, K)
            neighbors_xyzlab = torch.gather(extended_rawpoints_xyzlab, 2, extended_idx.cuda())  # shape (B, 3, N, K)

            extended_coords_xyz = extended_coords_xyzlab[:,:3,:,:]
            neighbors_xyz = neighbors_xyzlab[:,:3,:,:]
            relative_xyz = extended_coords_xyz - neighbors_xyz
            relative_dis = torch.sqrt(torch.sum(torch.pow(relative_xyz, 2), axis=1, keepdims=False))
            extended_coords_lab = extended_coords_xyzlab[:,3:,:,:]
            extended_coords_lab = extended_coords_lab.transpose(1,2)
            extended_coords_lab = extended_coords_lab.transpose(-2,-1)
            neighbors_lab = neighbors_xyzlab[:,3:,:,:]
            neighbors_lab = neighbors_lab.transpose(1,2)
            neighbors_lab = neighbors_lab.transpose(-2,-1)

            if not self.use_rgb :
                color_diff = ciede2000.CIEDE2000(extended_coords_lab,neighbors_lab) # b,n,k,1
                color_diff = color_diff.transpose(-2, -1).transpose(1, 2)
            else:
                relative_rgb = extended_coords_lab - neighbors_lab
                color_diff = torch.sqrt(torch.sum(torch.pow(relative_rgb, 2), axis=-1, keepdims=False)) # b,n,k


            # relative point position encoding
            concat = torch.cat((
                extended_coords_xyzlab,
                neighbors_xyzlab,
                extended_coords_xyzlab - neighbors_xyzlab,
                relative_dis.unsqueeze(-3),
                color_diff.unsqueeze(-3),
            ), dim=-3).to(self.device)# 6+6+6+1+1 = 20
            concat = self.mlp0(concat)
            concat = self.mlp1(concat)
            ste_feature = concat
        else:
            concat = ste_feature

        x =  torch.cat((
            concat, # B,8,n,k
            features.expand(B, -1, N, K)
        ), dim=-3)

        return x,ste_feature

class AttentivePooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentivePooling, self).__init__()

        self.score_fn = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.Softmax(dim=-2)
        )
        self.mlp = SharedMLP(in_channels, out_channels, bn=True, activation_fn=nn.ReLU())
        self.mlp_1 = SharedMLP(out_channels, out_channels)

    def forward(self, x, frame_id=None):
        r"""
            Forward pass

            Parameters
            ----------
            x: torch.Tensor, shape (B, d_in, N, K)

            Returns
            -------
            torch.Tensor, shape (B, d_out, N, 1)
        """
        # computing attention scores
        scores = self.score_fn(x.permute(0,2,3,1)).permute(0,3,1,2)
        # sum over the neighbors
        features = torch.sum(scores * x, dim=-1, keepdim=True) # shape (B, d_in, N, 1)
        features = self.mlp(features)
        features = self.mlp_1(features)
        return features

class LocalFeatureAggregation(nn.Module):

    def __init__(self, d_in, d_out, num_neighbors, device,use_rgb=False):
        super(LocalFeatureAggregation, self).__init__()

        self.num_neighbors = num_neighbors


        self.mlp1 = SharedMLP(d_in, d_out//2, activation_fn=nn.LeakyReLU(0.2))
        self.mlp1_1 = SharedMLP(d_out//2, d_out//2)
        self.mlp2 = SharedMLP(d_out*2, d_out*2)
        self.mlp2_2 = SharedMLP(d_out*2, d_out*2)

        self.lse1 = LocalSpatialEncoding(d_out//2, num_neighbors, device,use_rgb)
        self.lse2 = LocalSpatialEncoding(d_out//2, num_neighbors, device, use_rgb)
        self.lse3 = LocalSpatialEncoding(d_out // 2, num_neighbors, device, use_rgb)
        self.pool1 = AttentivePooling(d_out, d_out//2)
        self.pool2 = AttentivePooling(d_out, d_out)

        self.pool3 = AttentivePooling(24, d_out*2)
        self.shortcut0 = SharedMLP(d_out//2, 2 * d_out, bn=True)
        self.shortcut1 = SharedMLP(d_out, 2 * d_out, bn=True)
        self.lrelu = nn.LeakyReLU()

    def forward(self, coords, raw_points, neigh_idx,frame_id=None):
        r"""
            Forward pass

            Parameters
            ----------
            coords: torch.Tensor, shape (B, N, 6)
                coordinates of the non ground point cloud
            raw_points : torch.Tensor, shape (B, M, 6) M>>N
                coordinates of the all point cloud
            features: torch.Tensor, shape (B, d_in, N, 1)
                features of the point cloud

            Returns
            -------
            torch.Tensor, shape (B, 2*d_out, N, 1)
        """

        features = coords
        features = features.permute(0,2,1).unsqueeze(-1)
        x = self.mlp1(features)  #B,N,K,6 -- B,n,k,8
        x = self.mlp1_1(x)

        x,ste = self.lse1(coords,raw_points, x, neigh_idx) #
        x = self.pool1(x,frame_id=frame_id)

        sc0 = self.shortcut0(x)

        x,_ = self.lse2(coords, raw_points, x, neigh_idx,whether_cal=0,ste_feature=ste)
        x = self.pool2(x, frame_id=frame_id)

        sc1 = self.shortcut1(x)

        x,_ = self.lse3(coords, raw_points, x, neigh_idx, whether_cal=0,ste_feature=ste)
        x = self.pool3(x, frame_id=frame_id)

        x = self.mlp2(x)
        x = self.mlp2_2(x)

        return self.lrelu(x + sc0 + sc1)

class PV_ENcoNet_POOLING_SC3(nn.Module): # 中层与深层shortcut

    def __init__(self, model_cfg,**kwargs):
        super(PV_ENcoNet_POOLING_SC3, self).__init__()
       
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = self.device
        self.model_cfg = model_cfg
        
        self.num_neighbors = self.model_cfg.NUM_NEIGHBORS
        self.decimation = self.model_cfg.DECIMATION
        self.d_in = self.model_cfg.D_IN

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.use_rgb = self.model_cfg.USE_RGB

        self.use_nbg = self.model_cfg.USE_NBG
        self.randla_encoder = LocalFeatureAggregation(self.d_in, 16, self.num_neighbors, device,self.use_rgb)
         ## (B,N,32,1)

        self.sparse_shape = [41, 1600, 1408]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(38, 32, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(32),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(32, 32,3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(32, 48, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(48, 48, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(48, 48, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(48, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )


        self.device = device

        self.num_point_features = 128

        self = self.to(device)

    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features

    def forward(self, batch_dict):
        """
            Forward pass
            Parameters
            ----------
            input: torch.Tensor, shape (B, N, d_in)
                input points

            Returns
            -------
            torch.Tensor, shape (B, num_classes, N)
                segmentation scores for each point
        """

        start = time.time()
        neigh_idx = batch_dict['neighbor']
        voxel_features_batch = batch_dict['voxel_features_batch']
        voxel_coords_batch = batch_dict['voxel_coords_batch']
        frame_id = batch_dict['frame_id']

        if self.use_nbg:
            raw_points_batch = batch_dict['raw_points_batch']
            voxel_features_encode_batch = self.randla_encoder(voxel_features_batch, raw_points_batch, neigh_idx,
                                                              frame_id=frame_id)
        else:
            voxel_features_encode_batch = self.randla_encoder(voxel_features_batch, voxel_features_batch, neigh_idx,
                                                              frame_id=frame_id)
        t1 =time.time()
        '''         
        path = '/media/ddd/data2/3d_MOTS_Ex./Code/OpenPCDet-RandlaNet/tools/time/'
        if os.path.exists(path+'PointEncoding.npy'):
        	t = np.load(path+'PointEncoding.npy')
        	t = list(t)
        	t.append(t1-start)
        	#print(t)
        	np.save(path+'PointEncoding.npy',t)
        else:
        	np.save(path+'PointEncoding.npy',[t1-start])
        '''         	
        voxel_features_encode_batch = voxel_features_encode_batch.squeeze(-1).permute(0, 2, 1)
        voxel_features_encode_batch = torch.cat((voxel_features_batch,voxel_features_encode_batch),dim=-1) #6+32
        voxel_features_encode_batch_shape = voxel_features_encode_batch.shape
        voxel_features_encode_batch = voxel_features_encode_batch.reshape((-1,voxel_features_encode_batch_shape[-1]))
        batch_dict['voxel_features'] = voxel_features_encode_batch
        voxel_coords_batch_shape = voxel_coords_batch.shape
        batch_dict['voxel_coords'] = voxel_coords_batch.reshape((-1,voxel_coords_batch_shape[-1]))
        t0_input =time.time()
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        #print(voxel_features.dtype)
        batch_size = batch_dict['batch_size']
        #torch.backends.cudnn.enabled = True
        #torch.backends.cudnn.benchmark = True
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features.cuda(),
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        #t1_input =time.time()
        #print('input time:',t1_input-t0_input)
        #t0_input =time.time()
        x = self.conv_input(input_sp_tensor)
        #print(x.features.dtype)
        #t1_input =time.time()
        x_conv1 = self.conv1(x)
        #t1_conv1 =time.time()
        x_conv2 = self.conv2(x_conv1)
        #t1_conv2 =time.time()
        x_conv3 = self.conv3(x_conv2)
        #t1_conv3 =time.time()
        x_conv4 = self.conv4(x_conv3)
        #t1_conv4 =time.time()
        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)
        #t1_out =time.time()
        batch_dict['encoded_spconv_tensor'] = out
        batch_dict['encoded_spconv_tensor_stride'] = 8
        #torch.backends.cudnn.enabled = False
        #torch.backends.cudnn.benchmark = False
        t2 = time.time()
        '''         
        if os.path.exists(path+'VoxelEncoding.npy'):
        	t = np.load(path+'VoxelEncoding.npy')
        	t = list(t)
        	t.append(t2-t0_input)
        	np.save(path+'VoxelEncoding.npy',t)
        else:
        	np.save(path+'VoxelEncoding.npy',[t2-t0_input])
      
        if os.path.exists(path+'convinput.npy'):
        	t = np.load(path+'convinput.npy')
        	t = list(t)
        	t.append(t1_input-t0_input)
        	np.save(path+'convinput.npy',t)
        else:
        	np.save(path+'convinput.npy',[t1_input-t0_input])

        if os.path.exists(path+'conv1.npy'):
        	t = np.load(path+'conv1.npy')
        	t = list(t)
        	t.append(t1_conv1-t1_input)
        	np.save(path+'conv1.npy',t)
        else:
        	np.save(path+'conv1.npy',[t1_conv1-t1_input])

        if os.path.exists(path+'conv2.npy'):
        	t = np.load(path+'conv2.npy')
        	t = list(t)
        	t.append(t1_conv2-t1_conv1)
        	np.save(path+'conv2.npy',t)
        else:
        	np.save(path+'conv2.npy',[t1_conv2-t1_conv1])

        if os.path.exists(path+'conv3.npy'):
        	t = np.load(path+'conv3.npy')
        	t = list(t)
        	t.append(t1_conv3-t1_conv2)
        	np.save(path+'conv3.npy',t)
        else:
        	np.save(path+'conv3.npy',[t1_conv3-t1_conv2])


        if os.path.exists(path+'conv4.npy'):
        	t = np.load(path+'conv4.npy')
        	t = list(t)
        	t.append(t1_conv4-t1_conv3)
        	np.save(path+'conv4.npy',t)
        else:
        	np.save(path+'conv4.npy',[t1_conv4-t1_conv3])


        if os.path.exists(path+'convout.npy'):
        	t = np.load(path+'convout.npy')
        	t = list(t)
        	t.append(t1_out-t1_conv4)
        	np.save(path+'convout.npy',t)
        else:
        	np.save(path+'convout.npy',[t1_out-t1_conv4])
 
        '''
        #print('Point Encoding time:',t1-start)
        #print('Voxel Encoding time:',t2-t1)
        #print('	convinput time:',t1_input-t0_input)
        #print('	conv1 time:',t1_conv1-t1_input)
        #print('	conv2 time:',t1_conv2-t1_conv1)
        #print('	conv3 time:',t1_conv3-t1_conv2)
        #print('	conv4 time:',t1_conv4-t1_conv3)
        #print('	convout time:',t1_out-t1_conv4)
        return batch_dict


