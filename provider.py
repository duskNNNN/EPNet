import numpy as np
import torch
import random
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingWarmRestarts,StepLR,CosineAnnealingLR
def normalize_data(batch_data):
    """ Normalize the batch data, use coordinates of the block centered at origin,
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    B, N, C = batch_data.shape
    normal_data = np.zeros((B, N, C))
    for b in range(B):
        pc = batch_data[b]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        normal_data[b] = pc
    return normal_data

def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx

def shuffle_points(batch_data):
    """ Shuffle orders of points in each point cloud -- changes FPS behavior.
        Use the same shuffling idx for the entire batch.
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    idx = np.arange(batch_data.shape[1])
    np.random.shuffle(idx)
    return batch_data[:,idx,:]

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def rotate_point_cloud_z(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def rotate_point_cloud_with_normal(batch_xyz_normal):
    ''' Randomly rotate XYZ, normal point cloud.
        Input:
            batch_xyz_normal: B,N,6, first three channels are XYZ, last 3 all normal
        Output:
            B,N,6, rotated XYZ, normal point cloud
    '''
    for k in range(batch_xyz_normal.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_xyz_normal[k,:,0:3]
        shape_normal = batch_xyz_normal[k,:,3:6]
        batch_xyz_normal[k,:,0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        batch_xyz_normal[k,:,3:6] = np.dot(shape_normal.reshape((-1, 3)), rotation_matrix)
    return batch_xyz_normal

def rotate_perturbation_point_cloud_with_normal(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx6 array, original batch of point clouds and point normals
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))
        shape_pc = batch_data[k,:,0:3]
        shape_normal = batch_data[k,:,3:6]
        rotated_data[k,:,0:3] = np.dot(shape_pc.reshape((-1, 3)), R)
        rotated_data[k,:,3:6] = np.dot(shape_normal.reshape((-1, 3)), R)
    return rotated_data

def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k,:,0:3]
        rotated_data[k,:,0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def rotate_point_cloud_by_angle_with_normal(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx6 array, original batch of point clouds with normal
          scalar, angle of rotation
        Return:
          BxNx6 array, rotated batch of point clouds iwth normal
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k,:,0:3]
        shape_normal = batch_data[k,:,3:6]
        rotated_data[k,:,0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        rotated_data[k,:,3:6] = np.dot(shape_normal.reshape((-1,3)), rotation_matrix)
    return rotated_data

def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data

def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for batch_index in range(B):
        batch_data[batch_index,:,:] += shifts[batch_index,:]
    return batch_data

def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index,:,:] *= scales[batch_index]
    return batch_data

def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            batch_pc[b,drop_idx,:] = batch_pc[b,0,:] # set to the first point
    return batch_pc

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_optimizer(cfg,classifier):
    if cfg["optimizer"] == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=float(cfg["learning_rate"]),
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=float(cfg["decay_rate"])
        )
    elif cfg["optimizer"]  == 'Sgd':
        optimizer = torch.optim.SGD(classifier.parameters(), lr=float(cfg["learning_rate"]), momentum=0.9)
    elif cfg["optimizer"] == 'AdamW':
        optimizer = torch.optim.AdamW(
            classifier.parameters(),
            lr=float(cfg["learning_rate"]),
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=float(cfg["decay_rate"]),
            amsgrad=False
        )
    return optimizer

class LinearCyclicLR(_LRScheduler):
    def __init__(self, optimizer, max_lr, min_lr, cycle_length, last_epoch=-1):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.cycle_length = cycle_length
        super(LinearCyclicLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        cycle_progress = (self.last_epoch % self.cycle_length) / self.cycle_length
        current_lr = self.max_lr - (self.max_lr - self.min_lr) * cycle_progress
        return [current_lr for _ in self.base_lrs]
def set_schedule(cfg,optimizer):
    if cfg["schedule"] == 'CosineAnnealingWarmRestarts':
        schedule = CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=2)
    elif cfg["schedule"] == 'StepLR':
        # schedule = StepLR(optimizer, step_size=10, gamma=0.9)
        schedule = StepLR(optimizer, step_size=10, gamma=float(cfg['gamma']))
    elif cfg["schedule"] == 'CosineAnnealingLR':
        schedule = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    elif cfg["schedule"] == 'LinearCyclicLR':
        schedule = LinearCyclicLR(optimizer, max_lr=0.001, min_lr=1e-6, cycle_length=50)
    
    return schedule

def shuffle_points_gpu(points):
    idx = torch.randperm(points.shape[1], device=points.device)
    points = points[:, idx, :]
    return points

def rotate_point_cloud_with_normal_gpu(points):
    rotation_angles = torch.rand(points.shape[0], device=points.device) * 2 * torch.pi
    cos_vals = torch.cos(rotation_angles)
    sin_vals = torch.sin(rotation_angles)
    
    rotation_matrices = torch.stack([
        torch.stack([cos_vals, torch.zeros_like(cos_vals), sin_vals], dim=-1),
        torch.stack([torch.zeros_like(cos_vals), torch.ones_like(cos_vals), torch.zeros_like(cos_vals)], dim=-1),
        torch.stack([-sin_vals, torch.zeros_like(cos_vals), cos_vals], dim=-1)
    ], dim=1)
    
    points[:, :, :3] = torch.matmul(points[:, :, :3], rotation_matrices.transpose(1, 2))
    points[:, :, 3:6] = torch.matmul(points[:, :, 3:6], rotation_matrices.transpose(1, 2))
    
    return points

def rotate_perturbation_point_cloud_with_normal_gpu(points, angle_sigma=0.06, angle_clip=0.18):
    angles = torch.clamp(angle_sigma * torch.randn((points.shape[0], 3), device=points.device), -angle_clip, angle_clip)
    Rx = torch.eye(3, device=points.device).repeat(points.shape[0], 1, 1)
    Ry = torch.eye(3, device=points.device).repeat(points.shape[0], 1, 1)
    Rz = torch.eye(3, device=points.device).repeat(points.shape[0], 1, 1)
    
    Rx[:, 1, 1] = torch.cos(angles[:, 0])
    Rx[:, 1, 2] = -torch.sin(angles[:, 0])
    Rx[:, 2, 1] = torch.sin(angles[:, 0])
    Rx[:, 2, 2] = torch.cos(angles[:, 0])
    
    Ry[:, 0, 0] = torch.cos(angles[:, 1])
    Ry[:, 0, 2] = torch.sin(angles[:, 1])
    Ry[:, 2, 0] = -torch.sin(angles[:, 1])
    Ry[:, 2, 2] = torch.cos(angles[:, 1])
    
    Rz[:, 0, 0] = torch.cos(angles[:, 2])
    Rz[:, 0, 1] = -torch.sin(angles[:, 2])
    Rz[:, 1, 0] = torch.sin(angles[:, 2])
    Rz[:, 1, 1] = torch.cos(angles[:, 2])
    
    R = Rz @ Ry @ Rx
    points[:, :, :3] = torch.matmul(points[:, :, :3], R.transpose(1, 2))
    points[:, :, 3:6] = torch.matmul(points[:, :, 3:6], R.transpose(1, 2))
    
    return points

def jitter_point_cloud_gpu(points, sigma=0.01, clip=0.05):
    jitter_noise = torch.clamp(sigma * torch.randn_like(points), -clip, clip)
    points += jitter_noise
    return points

def shift_point_cloud_gpu(points, shift_range=0.1):
    shifts = torch.empty((points.shape[0], 1, 3), device=points.device).uniform_(-shift_range, shift_range)
    points[:, :, :3] += shifts
    return points

def random_scale_point_cloud_gpu(points, scale_low=0.8, scale_high=1.25):
    scales = torch.empty((points.shape[0], 1, 1), device=points.device).uniform_(scale_low, scale_high)
    points[:, :, :3] *= scales
    return points

def random_point_dropout_gpu(points, max_dropout_ratio=0.125):
    B, N, _ = points.shape
    
    dropout_ratios = torch.rand(B, 1, device=points.device) * max_dropout_ratio
    drop_mask = (torch.rand(B, N, device=points.device) <= dropout_ratios).unsqueeze(-1)
    
    first_point = points[:, 0:1, :].expand(B, N, points.shape[2])
    points = torch.where(drop_mask, first_point, points)
    
    return points

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def pc_normalize_noscale(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    return pc