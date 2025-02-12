import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points

def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs): # 515->256->512->1024
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

class SelfAttentionWithResidual(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttentionWithResidual, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.BatchNorm1d(embed_dim)  

    def forward(self, x):
        # x shape: (B, N, C) -> (N, B, C) for multihead attention
        x_permuted = x.permute(2, 0, 1)  
        attn_output, _ = self.self_attention(x_permuted, x_permuted, x_permuted)   
        attn_output = attn_output.permute(1, 2, 0)
        x = self.norm(x + attn_output)
        return x

def nearest_point_stack(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx >= N
    group_idx[mask] = group_first[mask]

    return group_idx

class SetAbstraction(nn.Module):
    def __init__(self, npoint, nsample, in_channel, mlp):
        super(SetAbstraction, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.conv = nn.Conv2d(nsample,round(1/2*nsample),1)
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        # new_xyz, new_points = sample_and_group_new(self.npoint, self.nsample, xyz, points)
        B, N, C = xyz.shape
        S = self.npoint
        nsample = self.nsample
        fps_idx = farthest_point_sample(xyz, S) # [B, npoint, C]
        new_xyz = index_points(xyz, fps_idx)
        # idx = query_ball_point(radius, nsample, xyz, new_xyz)
        idx = nearest_point_stack(nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
        grouped_points = index_points(points, idx)
        grouped_xyz_selected, grouped_points_selected = select_topk_by_cosine_similarity(grouped_xyz, grouped_points,round(1/2 * nsample), new_xyz)
        grouped_xyz_norm = grouped_xyz_selected - new_xyz.view(B, S, 1, C)
        grouped_points = self.conv(grouped_points.permute(0,2,1,3)).permute(0,2,1,3)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs): # 515->256->512->1024
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

def sample_and_group_new(npoint, nsample, xyz, points):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    # idx = query_ball_point(radius, nsample, xyz, new_xyz)
    idx = nearest_point_stack(nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_points = index_points(points, idx)
    grouped_xyz_selected, grouped_points_selected = select_topk_by_cosine_similarity(grouped_xyz, grouped_points,round(1/2 * nsample), new_xyz)
    grouped_xyz_norm = grouped_xyz_selected - new_xyz.view(B, S, 1, C)
    new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]

    return new_xyz, new_points

def compute_normals(grouped_xyz):
    B, S, K, C = grouped_xyz.shape
    
    centroid = torch.mean(grouped_xyz, dim=2, keepdim=True)  # [B, S, 1, C]
    centered_grouped_xyz = grouped_xyz - centroid  # [B, S, K, C]
    
    cov_matrix = torch.matmul(centered_grouped_xyz.transpose(2, 3), centered_grouped_xyz) / K  # [B, S, C, C]
    
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)  # [B, S, C], [B, S, C, C]
    
    normals = eigenvectors[..., 0]  # [B, S, C]
    normals = normals.unsqueeze(2).expand(-1, -1, K, -1)  # [B, S, K, C]
    return normals

def compute_cosine_similarity(normals_a, normals_b):
    normals_a = F.normalize(normals_a, dim=-1)
    normals_b = F.normalize(normals_b, dim=-1)
    
    cosine_similarity = torch.sum(normals_a * normals_b, dim=-1)  # [B, S, K]
    return cosine_similarity

def select_topk_by_cosine_similarity_normal(grouped_xyz, grouped_points, k, new_xyz):
    normals_grouped = compute_normals(grouped_xyz)  # [B, S, K, C]
    normals_new = compute_normals(new_xyz.unsqueeze(2))  # [B, S, 1, C]
    cosine_similarity = compute_cosine_similarity(normals_grouped, normals_new)  # [B, S, K]
    topk_indices = torch.topk(cosine_similarity, k, dim=-1)[1]  # [B, S, k]
    selected_grouped_xyz = torch.gather(
        grouped_xyz,  # [B, S, K, C]
        2,
        topk_indices.unsqueeze(-1).expand(-1, -1, -1, grouped_xyz.shape[-1])  # [B, S, k, C]
    )
    selected_normals = torch.gather(
        normals_grouped,  # [B, S, K, C]
        2,
        topk_indices.unsqueeze(-1).expand(-1, -1, -1, normals_grouped.shape[-1])  # [B, S, k, C]
    )
    selected_grouped_points = torch.gather(
        grouped_points,  # [B, S, K, D]
        2,
        topk_indices.unsqueeze(-1).expand(-1, -1, -1, grouped_points.shape[-1])  # [B, S, k, D]
    )
    return selected_grouped_xyz, selected_grouped_points, selected_normals

class PointNetSetAbstractionMsgSelectNormal(nn.Module):
    def __init__(self, npoint, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsgSelectNormal, self).__init__()
        self.npoint = npoint
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3 + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i in range(len(self.nsample_list)):
            K = self.nsample_list[i]
            # group_idx = query_ball_point(radius, K, xyz, new_xyz)
            group_idx = nearest_point_stack(K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_points = index_points(points, group_idx)
            grouped_xyz_selected, grouped_points_selected,selected_normals = select_topk_by_cosine_similarity_normal(grouped_xyz, grouped_points,round(3/4 * K), new_xyz)

            grouped_xyz_selected -= new_xyz.view(B, S, 1, C)
            grouped_points = torch.cat([grouped_points_selected, grouped_xyz_selected,selected_normals], dim=-1)

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat

def knn(xyz, new_xyz, k):
    """
    KNN implementation that returns the indices of k nearest neighbors.
    Input:
        xyz: all points, [B, N, 3]
        new_xyz: sampled points, [B, S, 3]
        k: number of nearest neighbors
    Return:
        idx: indices of k nearest neighbors, [B, S, K]
    """
    sqrdists = square_distance(new_xyz, xyz)  # [B, S, N]
    _, idx = torch.topk(sqrdists, k, largest=False, dim=-1)  # Find K nearest neighbors
    return idx

def compute_scales(grouped_points):
    """
    Compute the three scales based on the nearest and farthest distances.
    Input:
        grouped_points: [B, S, K, 3] points corresponding to K neighbors for each sampled point
    Return:
        scales: [B, S, 3] three scales (s1, s2, s3)
    """
    # Compute pairwise distances between the sampled point and its neighbors
    dists = torch.norm(grouped_points - grouped_points[:, :, 0:1, :], dim=-1)  # [B, S, K]
    p, _ = torch.min(dists, dim=-1)  # nearest distance [B, S]
    q, _ = torch.max(dists, dim=-1)  # farthest distance [B, S]
    
    s1 = p + (q - p) / 3
    s2 = p + 2 * (q - p) / 3
    s3 = q
    
    scales = torch.stack([s1, s2, s3], dim=-1)  # [B, S, 3]
    return scales

def query_ball_point_radius(radius, grouped_points, new_xyz):
    """
    Input:
        radius: local region radius for each sampled point [B, S] (dynamic radius)
        grouped_points: points corresponding to K neighbors for each sampled point [B, S, K, 3]
        new_xyz: query points (sampled points), [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, variable number of neighbors] (dynamic neighbors per scale)
    """
    device = grouped_points.device
    B, S, K, _ = grouped_points.shape

    sqrdists = torch.sum((grouped_points - new_xyz.unsqueeze(2)) ** 2, dim=-1)  # [B, S, K]

    group_idx = []
    
    for b in range(B):
        batch_group_idx = []
        for s in range(S):
            valid_idx = torch.where(sqrdists[b, s] < radius[b, s] ** 2)[0] 
            if valid_idx.shape[0] == 0:  
                valid_idx = torch.tensor([0], device=device)
            batch_group_idx.append(valid_idx)
        group_idx.append(batch_group_idx)
    
    return group_idx

def query_ball_point_new(radius, grouped_points, new_xyz):
    """
    Input:
        radius: local region radius for each sampled point [B, S] (dynamic radius)
        grouped_points: points corresponding to K neighbors for each sampled point [B, S, K, 3]
        new_xyz: query points (sampled points), [B, S, 3]
    Return:
        group_idx: mask tensor indicating valid neighbors within the radius, [B, S, K]
    """
    sqrdists = torch.sum((grouped_points - new_xyz.unsqueeze(2)) ** 2, dim=-1)  # [B, S, K]

    mask = sqrdists < radius.unsqueeze(-1) ** 2  # [B, S, K]
    
    return mask  

def query_ball_point_random(radius, grouped_points, new_xyz,ratio=0.75):
    """
    Query ball points based on radius and randomly sample 3/4 of the valid neighbors.
    Input:
        radius: local region radius for each sampled point [B, S] (dynamic radius)
        grouped_points: points corresponding to K neighbors for each sampled point [B, S, K, 3]
        new_xyz: query points (sampled points), [B, S, 3]
    Return:
        mask: mask tensor indicating randomly sampled 3/4 of valid neighbors within the radius, [B, S, K]
    """
    device = grouped_points.device
    B, S, K, _ = grouped_points.shape
    sqrdists = torch.sum((grouped_points - new_xyz.unsqueeze(2)) ** 2, dim=-1)  # [B, S, K]
    mask = sqrdists < radius.unsqueeze(-1) ** 2  # [B, S, K] 
    valid_counts = mask.sum(dim=-1, keepdim=True) 
    num_sampled = torch.clamp((ratio * valid_counts).int(), min=1) 
    rand_idx = torch.argsort(torch.rand(B, S, K, device=device), dim=-1) 
    sampled_idx = torch.arange(K, device=device).unsqueeze(0).unsqueeze(0) < num_sampled 
    mask = mask & sampled_idx.scatter(2, rand_idx, sampled_idx)  
    return mask  

def query_ball_point_new_radius(radius, grouped_points, new_xyz, k):
    """
    Input:
        radius: local region radius for each sampled point [B, S] (dynamic radius)
        grouped_points: points corresponding to K neighbors for each sampled point [B, S, K, 3]
        new_xyz: query points (sampled points), [B, S, 3]
        k: desired number of neighbors to be sampled
    Return:
        group_idx: tensor of shape [B, S, k] indicating valid neighbors, padded if necessary.
    """
    B, S, K, _ = grouped_points.shape

    sqrdists = torch.sum((grouped_points - new_xyz.unsqueeze(2)) ** 2, dim=-1)  # [B, S, K]
    mask = sqrdists < radius.unsqueeze(-1) ** 2  # [B, S, K]
    valid_counts = mask.sum(dim=-1)  # [B, S]
    first_valid_idx = mask.float().argmax(dim=-1, keepdim=True)  # [B, S, 1]
    first_valid_idx = first_valid_idx.repeat(1, 1, k)  # [B, S, k]
    selected_idx = torch.arange(K, device=grouped_points.device).view(1, 1, K).repeat(B, S, 1)  # [B, S, K]
    mask = mask.unsqueeze(-1).expand(-1, -1, K)  # [B, S, K]
    padded_idx = torch.where(mask[:, :, :k], selected_idx[:, :, :k], first_valid_idx)  # [B, S, k]

    return padded_idx  # [B, S, k]

def knn_point(k, xyz, new_xyz):
    sqrdists = square_distance(new_xyz, xyz)  # [B, S, N]
    _, group_idx = torch.topk(sqrdists, k, dim=-1, largest=False, sorted=True)
    
    grouped_xyz = index_points(xyz, group_idx)  # [B, S, k, 3]
    return group_idx, grouped_xyz

def local_query_ball(radius, nsample, grouped_xyz):
    B, S, k, _ = grouped_xyz.shape
    center_xyz = grouped_xyz[:, :, 0:1, :] 
    dists = torch.sum((grouped_xyz - center_xyz) ** 2, dim=-1)  # [B, S, k]
    mask = dists <= radius ** 2
    group_idx = torch.arange(k, dtype=torch.long, device=grouped_xyz.device).view(1, 1, k).repeat(B, S, 1)
    group_idx[~mask] = k  
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].unsqueeze(-1).repeat(1, 1, nsample)
    mask = group_idx == k
    group_idx[mask] = group_first[mask]
    return group_idx

class FactorizedSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, factor=4):
        super(FactorizedSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.factor = factor
        self.query_proj = nn.Linear(embed_dim, embed_dim // factor)
        self.key_proj = nn.Linear(embed_dim, embed_dim // factor)
        self.value_proj = nn.Linear(embed_dim, embed_dim // factor)
        self.out_proj = nn.Linear(embed_dim // factor, embed_dim)
        self.norm = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        # x shape: (B, C, N)
        B, C, N = x.shape
        x = x.permute(2, 0, 1)  
        Q = self.query_proj(x)  # (N, B, C // factor)
        K = self.key_proj(x)    # (N, B, C // factor)
        V = self.value_proj(x)  # (N, B, C // factor)
        attn_weights = torch.einsum("nbe,nbe->be", Q, K) / (self.embed_dim // self.factor) ** 0.5
        attn_output = torch.einsum("be,nbe->nbe", attn_weights, V)
        attn_output = self.out_proj(attn_output)  # (N, B, C)
        attn_output = attn_output.permute(1, 2, 0)
        x = self.norm(x.permute(1, 2, 0) + attn_output)
        return x