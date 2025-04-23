import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
# from .improve import *
# from torch_cluster import fps
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

# 返回索引的点
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

# 最远点采样
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
    # 创建一个存储采样点索引的张量 centroids，初始为0
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    # 创建一个距离张量 distance，初始化为一个非常大的值 1e10，用于存储每个点到已选择的最近采样点的距离
    distance = torch.ones(B, N).to(device) * 1e10
    # 随机初始化一个最远点 farthest 的索引，表示每个批次中初始选中的点
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    # 创建一个表示批次索引的张量
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        # 将当前最远点的索引保存到 centroids 中第 i 列
        centroids[:, i] = farthest
        # 根据索引 farthest 获取当前最远点的坐标，并将其形状调整为 [B, 1, 3]
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        # 计算所有点到当前采样点（centroid）的欧氏距离的平方，并得到形状为 [B, N] 的距离矩阵 dist
        dist = torch.sum((xyz - centroid) ** 2, -1)
        # 创建一个布尔掩码 mask，表示哪些点到当前采样点的距离比它们之前记录的最近距离更小
        mask = dist < distance
        # 更新 distance，将那些距离更小的点的距离值更新为 dist 中相应的值
        distance[mask] = dist[mask]
        # 选择当前距离最远的那个点作为下一个采样点，并更新 farthest 索引
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
    # 按照距离排序截取前nsample个
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    # 不够nsample个时用第一个有效点进行补齐
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

# def sample_and_group_cluster(npoint, radius, nsample, xyz, points, returnfps=False):
#     """
#     Input:
#         npoint:
#         radius:
#         nsample:
#         xyz: input points position data, [B, N, 3]
#         points: input points data, [B, N, D]
#     Return:
#         new_xyz: sampled points position data, [B, npoint, nsample, 3]
#         new_points: sampled points data, [B, npoint, nsample, 3+D]
#     """
#     device = xyz.device
#     B, N, C = xyz.shape
#     S = npoint
#     fps_idx = fps(xyz.contiguous().view(-1, C).to(device), 
#                   batch=torch.arange(B).repeat_interleave(N).to(device), 
#                   ratio=None, random_start=False)[:S]
#     new_xyz = xyz[torch.arange(B).unsqueeze(1), fps_idx]
#     idx = nearest_point_stack(nsample, xyz, new_xyz)
#     grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
#     grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

#     if points is not None:
#         grouped_points = index_points(points, idx)
#         new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
#     else:
#         new_points = grouped_xyz_norm
#     if returnfps:
#         return new_xyz, new_points, grouped_xyz, fps_idx
#     else:
#         return new_xyz, new_points

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
        for i, conv in enumerate(self.mlp_convs): # 暴力从515->256->512->1024
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))
        # 对第3维进行最大化处理
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
        # 采样点
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            # 查询指定范围内的索引
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            # 将邻域点坐标中心化，减去对应的采样点坐标 new_xyz
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                # 通过索引 group_idx 从 points 中提取邻域点的特征信息
                grouped_points = index_points(points, group_idx)
                # 将邻域点的特征与坐标信息拼接在一起，形状为 [B, S, K, D+C]
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
        # 将 new_points_list 中的所有特征拼接在一起，形成最终的特征 new_points_concat，形状为 [B, D', S]
        new_points_concat = torch.cat(new_points_list, dim=1)
        # 返回采样后的点云坐标 和对应的特征 
        return new_xyz, new_points_concat

# 用于恢复分辨率
# 原理：距离越近的点，特征也是相似的
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
        # 点云上采样
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
            # 如果采样点集有多个点 (S > 1)，则需要进行插值计算
            # 计算 xyz1 中每个点到 xyz2 中各个点的平方距离
            dists = square_distance(xyz1, xyz2)
            # 按距离升序排序
            dists, idx = dists.sort(dim=-1)
            # 取距离最近的三个点，进行三线性插值，保留前3个最小距离及其对应索引
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
            # 计算距离的倒数，以便生成权重，1e-8 是为了避免除零错误
            dist_recip = 1.0 / (dists + 1e-8)
            # 计算权重的归一化因子 norm，用于归一化每个点的权重
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            # 生成插值权重 weight，使其总和为1
            weight = dist_recip / norm
            # 根据插值权重对 points2 中的最近3个点的特征进行加权求和，得到插值后的特征 interpolated_points
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

# 带残差连接的自注意力模块
class SelfAttentionWithResidual(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttentionWithResidual, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads)
        # 添加 LayerNorm 以增强稳定性
        self.norm = nn.BatchNorm1d(embed_dim)  

    def forward(self, x):
        # x shape: (B, N, C) -> (N, B, C) for multihead attention
        # 转换形状为 (N, B, C)
        x_permuted = x.permute(2, 0, 1)  
        
        # 自注意力和残差连接
        attn_output, _ = self.self_attention(x_permuted, x_permuted, x_permuted)
        
        # 转换回原始形状 (B, C, N)
        attn_output = attn_output.permute(1, 2, 0)
        
        # 添加残差连接和规范化
        x = self.norm(x + attn_output)
        
        return x

# 基于边的卷积操作，增加局部特征
class PointNetEdgeConvBlock(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channels, mlp_list):
        super(PointNetEdgeConvBlock, self).__init__()
        self.sa = PointNetSetAbstractionMsgLocalFeature(npoint, radius_list, nsample_list, in_channels, mlp_list)
        self.in_channels = in_channels
        self.out_channels = np.sum([x[-1] for x in mlp_list])
        edge_conv_channels = self.out_channels
        # EdgeConv block
        self.edge_conv = nn.Sequential(
            nn.Conv2d(2 * edge_conv_channels, edge_conv_channels, 1),
            nn.BatchNorm2d(edge_conv_channels),
            nn.ReLU(),
            nn.Conv2d(edge_conv_channels, edge_conv_channels, 1),
            nn.BatchNorm2d(edge_conv_channels),
            nn.ReLU(),
        )
        
        self.linear_transform = nn.Linear(self.in_channels + 3, self.out_channels)


    def forward(self, xyz, points):
        # 获取采样点和多尺度邻域信息
        new_xyz, new_points, all_grouped_xyz, all_grouped_points = self.sa(xyz, points)

        # 使用 EdgeConv 操作（示例：只对第一个尺度的邻域进行 EdgeConv）
        grouped_points = all_grouped_points[0]  # [B, S, K, D+3]
        # unsqueeze（2） 针对第二个维度
        new_points = new_points.permute(0, 2, 1).unsqueeze(2)
        grouped_points = self.linear_transform(grouped_points)
        grouped_points_diff = grouped_points - new_points  # [B, S, K, D+3]

        # 拼接原始特征和特征差异
        edge_features =  torch.cat([grouped_points, grouped_points_diff], dim=-1) # [B, S, K, 2*(D+3)]
        edge_features = edge_features.permute(0, 3, 2, 1)  # [B, 2*(D+3), K, S]

        # 应用 EdgeConv
        edge_features = self.edge_conv(edge_features)  # [B, D, K, S]

        # 聚合邻域特征（例如通过最大池化）
        new_points = torch.max(edge_features, dim=2)[0]  # [B, D, S]

        return new_xyz, new_points

class PointNetSetAbstractionCluster(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstractionCluster, self).__init__()
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
        for i, conv in enumerate(self.mlp_convs): # 暴力从515->256->512->1024
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))
        # 对第3维进行最大化处理
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

# 特征提取
class PointNetSetAbstractionMsgCluster(nn.Module):
    def __init__(self, npoint, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsgCluster, self).__init__()
        self.npoint = npoint
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        # self.se_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            # ses = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
                # ses.append(SEBlock(out_channel))
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)
            # self.se_blocks.append(ses)

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
        device = xyz.device
        B, N, C = xyz.shape
        S = self.npoint
        # 采样点
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, K in enumerate(self.nsample_list):
            # 查询指定范围内的索引
            group_idx = nearest_point_stack(K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            # 将邻域点坐标中心化，减去对应的采样点坐标 new_xyz
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                # 通过索引 group_idx 从 points 中提取邻域点的特征信息
                grouped_points = index_points(points,group_idx)
                # 将邻域点的特征与坐标信息拼接在一起，形状为 [B, S, K, D+C]
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)),inplace=True)
                # 添加通道注意力模块
                # grouped_points = self.se_blocks[i][j](grouped_points)

            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        # 将 new_points_list 中的所有特征拼接在一起，形成最终的特征 new_points_concat，形状为 [B, D', S]
        new_points_concat = torch.cat(new_points_list, dim=1)
        # 返回采样后的点云坐标 和对应的特征 
        return new_xyz, new_points_concat

# 特征传播
class PointNetFeaturePropagationCluster(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagationCluster, self).__init__()
        # self.mlp_convs = nn.ModuleList()
        self.mlp_linears = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_linears.append(nn.Linear(last_channel,out_channel,bias=False))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2,k=16):
        # 点云上采样
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
            # 如果采样点集有多个点 (S > 1)，则需要进行插值计算
            # 计算 xyz1 中每个点到 xyz2 中各个点的平方距离
            dists = square_distance(xyz1, xyz2)
            # 按距离升序排序
            # dists, idx = dists.sort(dim=-1)
            _, idx = dists.topk(k=k, dim=-1, largest=False, sorted=False)  # 选取前k个邻居
            # 取距离最近的三个点，进行三线性插值，保留前3个最小距离及其对应索引
            # dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
            # 计算距离的倒数，以便生成权重，1e-8 是为了避免除零错误
            dist_recip = 1.0 / (dists.gather(-1, idx) + 1e-8)
            # 计算权重的归一化因子 norm，用于归一化每个点的权重
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            # 生成插值权重 weight，使其总和为1
            weight = dist_recip / norm
            # 根据插值权重对 points2 中的最近3个点的特征进行加权求和，得到插值后的特征 interpolated_points
            interpolated_points = torch.sum(index_points(points2, idx) * weight.unsqueeze(-1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, linear in enumerate(self.mlp_linears):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(linear(new_points.permute(0,2,1)).permute(0,2,1)),inplace=True)
        return new_points

# 借用堆排序的思想，只对其几个进行排序
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

    # 计算每个查询点到所有点的欧氏距离平方
    sqrdists = square_distance(new_xyz, xyz)

    # 使用 topk 而非排序，只获取前 nsample 个最近的点
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)

    # 对于少于 nsample 个点的情况，使用第一个点进行填充
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx >= N
    group_idx[mask] = group_first[mask]

    return group_idx

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 对每个通道进行全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.BatchNorm1d(channel // reduction),  # 添加BatchNorm1d
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # 对输入 x 进行全局平均池化，输出形状为 [B, C]
        y = self.fc(y).view(b, c, 1, 1)  # 生成通道权重，形状为 [B, C, 1, 1]
        
        # 注意力机制：对原始输入特征进行加权
        weighted_x = x * y.expand_as(x)  # 注意：将注意力权重应用到每个通道
        
        # 残差连接：将加权后的特征与原始输入相加
        out = weighted_x + x  # 残差连接
        
        return out

# 用knn替换球状采样，邻居点设为8,16,32
# 随机丢弃原始点，与模块后的结果做一次自注意力  
class PointNetSetAbstractionMsg_MultiKnn_SelfAttention(nn.Module):
    def __init__(self,npoints,sample_points, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg_MultiKnn_SelfAttention, self).__init__()
        self.npoints = npoints
        self.radius_list = radius_list
        self.sample_points = sample_points
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        self.out_channel = sum(sublist[-1] for sublist in mlp_list)
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
        self.sa = SelfAttentionFusion(6,self.out_channel)

    def forward(self, xyz, points,global_points):
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
        S = self.sample_points
        # 采样点
        indices = farthest_point_sample(xyz, S)
        new_xyz = index_points(xyz,indices)
        # points_sample = index_points(points,indices)
        # # 做一次自注意力
        # indices = torch.randint(0, self.npoints, (self.sample_points,)) 
        # global_points_random = global_points[:, :, indices].permute(0, 2, 1)
        # # new_points_concat = self.sa(global_points_random,points_sample)
        # points = torch.cat((global_points_random,points_sample),dim=1)
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            # 查询指定范围内的索引
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            # 将邻域点坐标中心化，减去对应的采样点坐标 new_xyz
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                # 通过索引 group_idx 从 points 中提取邻域点的特征信息
                grouped_points = index_points(points, group_idx)
                # 将邻域点的特征与坐标信息拼接在一起，形状为 [B, S, K, D+C]
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
        # 将 new_points_list 中的所有特征拼接在一起，形成最终的特征 new_points_concat，形状为 [B, D', S]
        new_points_concat = torch.cat(new_points_list, dim=1)
        # 返回采样后的点云坐标 和对应的特征 
        return new_xyz, new_points_concat

class SelfAttentionFusion(nn.Module):
    def __init__(self, in_channel_g, in_channel_l):
        super(SelfAttentionFusion, self).__init__()
        
        # 查询、键、值矩阵的线性变换 (1x1 卷积)
        self.query_conv = nn.Conv1d(in_channel_g + in_channel_l, 2 * (in_channel_g + in_channel_l), 1)
        self.key_conv = nn.Conv1d(in_channel_g + in_channel_l, 2 * (in_channel_g + in_channel_l), 1)
        self.value_conv = nn.Conv1d(in_channel_g + in_channel_l, 2 * (in_channel_g + in_channel_l), 1)
        
        # 输出变换
        self.out_conv1 = nn.Conv1d(2 * (in_channel_g + in_channel_l), in_channel_l, 1)
        self.bn1 = nn.BatchNorm1d(in_channel_l)


    def forward(self, F_g, F_l):
        """
        输入：
        - F_g: 全局特征, 形状为 (B, in_channel_g, N)
        - F_l: 局部特征, 形状为 (B, in_channel_l, N)
        
        输出：
        - F_output: 融合后的特征，形状为 (B, in_channel_l, N)
        """
        # 将全局和局部特征拼接在一起形成 features
        features = torch.cat([F_g, F_l], dim=1)  # (B, in_channel_g + in_channel_l, N)
        
        # 计算查询、键、值矩阵
        Q = self.query_conv(features)  # (B, 2 * (in_channel_g + in_channel_l), N)
        K = self.key_conv(features)    # (B, 2 * (in_channel_g + in_channel_l), N)
        V = self.value_conv(features)  # (B, 2 * (in_channel_g + in_channel_l), N)

        # 转置 Q 和 K，计算注意力权重 A
        attention = torch.bmm(Q.transpose(1, 2), K)  # (B, N, N)
        attention = attention / (K.shape[1] ** 0.5)  # 缩放
        attention = torch.softmax(attention, dim=-1) # 对每个点进行 softmax 归一化
        
        # 转置 V，使其匹配注意力矩阵的乘法要求
        V = V.permute(0, 2, 1)  # (B, N, 2 * (in_channel_g + in_channel_l))

        # 加权值矩阵 V，得到融合特征
        F_fused = torch.bmm(attention, V)  # (B, N, 2 * (in_channel_g + in_channel_l))
        F_fused = F_fused.permute(0, 2, 1)  # (B, 2 * (in_channel_g + in_channel_l), N)
        
        # 输出变换，使其回到原始通道维度
        F_output = F.relu(self.bn1(self.out_conv1(F_fused)),inplace=True)  
        
        return F_output

# 注意力机制模块定义
class AttentionBasedPointAggregation(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64):
        super(AttentionBasedPointAggregation, self).__init__()
        # 将邻域点的坐标 (x, y, z) 映射到高维特征
        self.linear = nn.Linear(input_dim, hidden_dim)
        # 计算每个邻域点的注意力分数
        self.attention_fc = nn.Linear(hidden_dim, 1)

    def forward(self, grouped_xyz):
        # 将邻域点 (x, y, z) 坐标映射到高维特征空间
        grouped_features = self.linear(grouped_xyz)  # 输出形状 (B, S, K, hidden_dim)

        # 计算每个邻域点的注意力分数
        attention_scores = self.attention_fc(grouped_features)  # 输出形状 (B, S, K, 1)

        # 通过 softmax 对邻域点的注意力权重进行归一化
        attention_weights = F.softmax(attention_scores, dim=2)  # 形状 (B, S, K, 1)

        # 使用注意力权重对邻域点特征进行加权聚合
        attended_features = torch.sum(attention_weights * grouped_features, dim=2)  # 输出形状 (B, S, hidden_dim)

        return attended_features

# PointNet Set Abstraction MSG with Attention
class PointNetSetAbstractionMsgMaxAttention(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsgMaxAttention, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()

        # 添加 AttentionBasedPointAggregation 模块
        self.attention_modules = nn.ModuleList()

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

            # 为每个尺度添加 Attention 模块
            self.attention_modules.append(AttentionBasedPointAggregation(input_dim=last_channel, hidden_dim=last_channel))

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
        # 采样点
        new_xyz = index_points_resize(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            # 查询指定范围内的索引
            group_idx = query_ball_point_resize(radius, K, xyz, new_xyz)
            grouped_xyz = index_points_resize(xyz, group_idx)
            # 将邻域点坐标中心化，减去对应的采样点坐标 new_xyz
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                # 通过索引 group_idx 从 points 中提取邻域点的特征信息
                grouped_points = index_points_resize(points, group_idx)
                # 将邻域点的特征与坐标信息拼接在一起，形状为 [B, S, K, D+C]
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))

            # 使用注意力机制代替 max pooling 进行特征聚合
            # 首先将维度转换回 [B, S, K, D]，以便 Attention 模块处理
            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, S, K, D]
            
            # 使用注意力模块进行聚合
            new_points = self.attention_modules[i](grouped_points)  # [B, S, hidden_dim]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        # 将 new_points_list 中的所有特征拼接在一起，形成最终的特征 new_points_concat，形状为 [B, D', S]
        new_points_concat = torch.cat(new_points_list, dim=2).permute(0, 2, 1)  # 拼接在特征维度 [B, D', S]

        return new_xyz, new_points_concat

# 类似单层的msg
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
        for i, conv in enumerate(self.mlp_convs): # 暴力从515->256->512->1024
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))
        # 对第3维进行最大化处理
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

# 用KNN替代球状采样
# 找到最近的N个点，计算其的法线与采样点法线的余弦相似度，取1/2*N个点
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
    """
    计算每个邻域点的法线。
    Args:
        grouped_xyz: 邻域点的坐标，形状为 [B, S, K, C]
    Returns:
        normals: 每个邻域点的法线，形状为 [B, S, K, C]
    """
    B, S, K, C = grouped_xyz.shape
    
    # 去中心化邻域点
    centroid = torch.mean(grouped_xyz, dim=2, keepdim=True)  # [B, S, 1, C]
    centered_grouped_xyz = grouped_xyz - centroid  # [B, S, K, C]
    
    # 计算每个邻域的协方差矩阵
    cov_matrix = torch.matmul(centered_grouped_xyz.transpose(2, 3), centered_grouped_xyz) / K  # [B, S, C, C]
    
    # 计算协方差矩阵的特征向量和特征值
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)  # [B, S, C], [B, S, C, C]
    
    # 最小特征值对应的特征向量即为法线方向
    normals = eigenvectors[..., 0]  # [B, S, C]，最小特征值对应的特征向量
    normals = normals.unsqueeze(2).expand(-1, -1, K, -1)  # [B, S, K, C]，扩展到每个邻域点
    return normals

def compute_cosine_similarity(normals_a, normals_b):
    """
    计算两个法线之间的余弦相似度。
    Args:
        normals_a: 邻域点的法线，形状为 [B, S, K, C]
        normals_b: 采样点的法线，形状为 [B, S, 1, C]
    Returns:
        cosine_similarity: 余弦相似度，形状为 [B, S, K]
    """
    # 归一化法线
    normals_a = F.normalize(normals_a, dim=-1)
    normals_b = F.normalize(normals_b, dim=-1)
    
    # 计算余弦相似度
    cosine_similarity = torch.sum(normals_a * normals_b, dim=-1)  # [B, S, K]
    return cosine_similarity

# 不返回法线信息
def select_topk_by_cosine_similarity(grouped_xyz, grouped_points, k, new_xyz):
    """
    根据余弦相似度选择邻域中的前k个点。
    Args:
        grouped_xyz: 邻域点的坐标，形状为 [B, S, K, C]
        grouped_points: 邻域点的特征，形状为 [B, S, K, D]
        k: 需要选择的点的数量
        new_xyz: 采样点坐标，形状为 [B, S, C]
    Returns:
        selected_grouped_xyz: 选择后的坐标，形状为 [B, S, k, C]
        selected_grouped_points: 选择后的特征，形状为 [B, S, k, D]
    """
    # 计算邻域点和采样点的法线
    normals_grouped = compute_normals(grouped_xyz)  # [B, S, K, C]
    normals_new = compute_normals(new_xyz.unsqueeze(2))  # [B, S, 1, C]
    
    # 计算余弦相似度
    cosine_similarity = compute_cosine_similarity(normals_grouped, normals_new)  # [B, S, K]
    
    # 对相似度排序，选择前k个
    topk_indices = torch.topk(cosine_similarity, k, dim=-1)[1]  # [B, S, k]
    
    # 根据索引选择坐标和特征
    # 使用`torch.gather`选择坐标
    selected_grouped_xyz = torch.gather(
        grouped_xyz,  # [B, S, K, C]
        2,
        topk_indices.unsqueeze(-1).expand(-1, -1, -1, grouped_xyz.shape[-1])  # [B, S, k, C]
    )

    # 如果还需要选择特征
    selected_grouped_points = torch.gather(
        grouped_points,  # [B, S, K, D]
        2,
        topk_indices.unsqueeze(-1).expand(-1, -1, -1, grouped_points.shape[-1])  # [B, S, k, D]
    )
    return selected_grouped_xyz, selected_grouped_points

# 返回法线信息，增加xyz的特征
def select_topk_by_cosine_similarity_normal(grouped_xyz, grouped_points, k, new_xyz):
    """
    根据余弦相似度选择邻域中的前k个点，并返回其法线。
    Args:
        grouped_xyz: 邻域点的坐标，形状为 [B, S, K, C]
        grouped_points: 邻域点的特征，形状为 [B, S, K, D]
        k: 需要选择的点的数量
        new_xyz: 采样点坐标，形状为 [B, S, C]
    Returns:
        selected_grouped_xyz: 选择后的坐标，形状为 [B, S, k, C]
        selected_grouped_points: 选择后的特征，形状为 [B, S, k, D]
        selected_normals: 选择后的法线，形状为 [B, S, k, C]
    """
    # 计算邻域点和采样点的法线
    normals_grouped = compute_normals(grouped_xyz)  # [B, S, K, C]
    normals_new = compute_normals(new_xyz.unsqueeze(2))  # [B, S, 1, C]
    
    # 计算余弦相似度
    cosine_similarity = compute_cosine_similarity(normals_grouped, normals_new)  # [B, S, K]
    
    # 对相似度排序，选择前k个
    topk_indices = torch.topk(cosine_similarity, k, dim=-1)[1]  # [B, S, k]
    
    # 根据索引选择坐标和特征
    # 使用`torch.gather`选择坐标
    selected_grouped_xyz = torch.gather(
        grouped_xyz,  # [B, S, K, C]
        2,
        topk_indices.unsqueeze(-1).expand(-1, -1, -1, grouped_xyz.shape[-1])  # [B, S, k, C]
    )

    # 选择对应的法线
    selected_normals = torch.gather(
        normals_grouped,  # [B, S, K, C]
        2,
        topk_indices.unsqueeze(-1).expand(-1, -1, -1, normals_grouped.shape[-1])  # [B, S, k, C]
    )

    # 如果还需要选择特征
    selected_grouped_points = torch.gather(
        grouped_points,  # [B, S, K, D]
        2,
        topk_indices.unsqueeze(-1).expand(-1, -1, -1, grouped_points.shape[-1])  # [B, S, k, D]
    )

    return selected_grouped_xyz, selected_grouped_points, selected_normals

class PointNetSetAbstractionMsgSelect(nn.Module):
    def __init__(self, npoint, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsgSelect, self).__init__()
        self.npoint = npoint
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
        # 采样点
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i in range(len(self.nsample_list)):
            K = self.nsample_list[i]
            # 查询指定范围内的索引
            # group_idx = query_ball_point(radius, K, xyz, new_xyz)
            group_idx = nearest_point_stack(K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_points = index_points(points, group_idx)
            grouped_xyz_selected, grouped_points_selected = select_topk_by_cosine_similarity(grouped_xyz, grouped_points,round(3/4 * K), new_xyz)

            # 将邻域点坐标中心化，减去对应的采样点坐标 new_xyz
            grouped_xyz_selected -= new_xyz.view(B, S, 1, C)
            grouped_points = torch.cat([grouped_points_selected, grouped_xyz_selected], dim=-1)

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        # 将 new_points_list 中的所有特征拼接在一起，形成最终的特征 new_points_concat，形状为 [B, D', S]
        new_points_concat = torch.cat(new_points_list, dim=1)
        # 返回采样后的点云坐标 和对应的特征 
        return new_xyz, new_points_concat

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
        # 采样点
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i in range(len(self.nsample_list)):
            K = self.nsample_list[i]
            # 查询指定范围内的索引
            # group_idx = query_ball_point(radius, K, xyz, new_xyz)
            group_idx = nearest_point_stack(K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_points = index_points(points, group_idx)
            grouped_xyz_selected, grouped_points_selected,selected_normals = select_topk_by_cosine_similarity_normal(grouped_xyz, grouped_points,round(3/4 * K), new_xyz)

            # 将邻域点坐标中心化，减去对应的采样点坐标 new_xyz
            grouped_xyz_selected -= new_xyz.view(B, S, 1, C)
            # 拼接上法线信息
            grouped_points = torch.cat([grouped_points_selected, grouped_xyz_selected,selected_normals], dim=-1)

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        # 将 new_points_list 中的所有特征拼接在一起，形成最终的特征 new_points_concat，形状为 [B, D', S]
        new_points_concat = torch.cat(new_points_list, dim=1)
        # 返回采样后的点云坐标 和对应的特征 
        return new_xyz, new_points_concat

class PointNetSetAbstractionMsgKNN(nn.Module):
    def __init__(self, npoint, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsgKNN, self).__init__()
        self.npoint = npoint
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
        # 采样点
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i in range(len(self.nsample_list)):
            K = self.nsample_list[i]
            # 查询指定范围内的索引
            group_idx = nearest_point_stack(K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_points = index_points(points, group_idx)

            # 将邻域点坐标中心化，减去对应的采样点坐标 new_xyz
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        # 将 new_points_list 中的所有特征拼接在一起，形成最终的特征 new_points_concat，形状为 [B, D', S]
        new_points_concat = torch.cat(new_points_list, dim=1)
        # 返回采样后的点云坐标 和对应的特征 
        return new_xyz, new_points_concat

def convert_pointcloud(cloud, n):
    # cloud: 输入的点云 tensor，形状为 (4, 3, 2048)
    batch_size, coord_dim, num_points = cloud.shape
    
    # Step 1: 计算每个点的基础信息，即每个点的x^2 + y^2 + z^2
    base_info = torch.sum(cloud ** 2, dim=1)  # 结果形状为 (4, 2048)
    
    # Step 2: 对基础信息计算每个批次的最大值和最小值，并为每个批次划分 n 个区间
    min_info = base_info.min(dim=1, keepdim=True)[0]  # 形状为 (4, 1)
    max_info = base_info.max(dim=1, keepdim=True)[0]  # 形状为 (4, 1)
    
    # 为每个批次生成 n + 1 个边界值
    bins = torch.linspace(0, 1, n + 1, device=cloud.device).unsqueeze(0)  # (1, n+1)
    bins = min_info + bins * (max_info - min_info)  # 形状为 (4, n+1)，对每个批次进行线性插值生成 n 个区间
    
    # Step 3: 使用广播机制找到每个点属于哪个区间
    # base_info: (4, 2048) -> (4, 2048, 1), bins: (4, n+1) -> (4, 1, n+1)
    base_info_expanded = base_info.unsqueeze(-1)
    bin_diff = base_info_expanded - bins.unsqueeze(1)  # (4, 2048, n+1)
    
    # 对每个点找到其落在哪个区间, 通过检测 bin_diff 的符号变化
    bin_indices = torch.sum(bin_diff > 0, dim=-1) - 1  # (4, 2048)，每个点对应的区间索引
    
    # Step 4: 随机生成编码，如果点属于区间，则生成0.6-0.9的随机数，不属于则生成0.1-0.4
    belong_values = torch.rand(batch_size, n, num_points, device=cloud.device).uniform_(0.6, 0.9)
    not_belong_values = torch.rand(batch_size, n, num_points, device=cloud.device).uniform_(0.1, 0.4)
    
    # 初始化编码矩阵
    encoded_cloud = torch.zeros(batch_size, n, num_points, device=cloud.device)
    
    # 创建 mask，判断每个点是否属于对应区间
    bin_indices_expanded = bin_indices.unsqueeze(1).expand(-1, n, -1)  # (4, n, 2048)
    range_tensor = torch.arange(n, device=cloud.device).view(1, -1, 1)  # (1, n, 1)
    
    # 判断属于哪个区间：bin_indices_expanded == range_tensor，生成对应的 mask
    belong_mask = bin_indices_expanded == range_tensor
    
    # 使用 mask 进行赋值
    encoded_cloud = torch.where(belong_mask, belong_values, not_belong_values)
    
    return torch.cat([cloud,encoded_cloud],dim=1)

def polynomial_mapping(coords, degree=4):
    """
    coords: (B, 3, N) 输入的 x, y, z 坐标，其中 B 是批次数，N 是点的数量
    degree: 多项式的最大阶数，支持扩展到4次、5次、6次
    返回多项式扩展后的形状，维度将取决于 degree
    """
    x, y, z = coords[:, 0, :], coords[:, 1, :], coords[:, 2, :]
    
    # 提前计算出常用的组合，避免重复计算
    xy = x + y
    xz = x + z
    yz = y + z
    xy_z = x + y - z
    x_yz = x - y + z
    x_y_z = x - y - z
    neg_x = -x
    neg_y = -y
    neg_z = -z
    
    # 准备收集所有特征
    features = []

    # 一次项
    if degree >= 1:
        # 直接计算一次项的8个组合
        features.extend([xy + z, xy - z, x_yz, x_y_z, neg_x + y + z, neg_x + y - z, neg_x + neg_y + z, neg_x + neg_y + neg_z])

    # 二次项
    if degree >= 2:
        # 预计算二次项，减少重复计算
        x2, y2, z2 = x ** 2, y ** 2, z ** 2
        features.extend([
            x2 + y2 + z2, x2 + y2 - z2, x2 - y2 + z2, x2 - y2 - z2,
            -x2 + y2 + z2, -x2 + y2 - z2, -x2 - y2 + z2, -x2 - y2 - z2
        ])

    # 三次项
    if degree >= 3:
        # 预计算三次项
        x3, y3, z3 = x ** 3, y ** 3, z ** 3
        features.extend([
            x3 + y3 + z3, x3 + y3 - z3, x3 - y3 + z3, x3 - y3 - z3,
            -x3 + y3 + z3, -x3 + y3 - z3, -x3 - y3 + z3, -x3 - y3 - z3
        ])

    # 四次项
    if degree >= 4:
        # 预计算四次项
        x4, y4, z4 = x ** 4, y ** 4, z ** 4
        features.extend([
            x4 + y4 + z4, x4 + y4 - z4, x4 - y4 + z4, x4 - y4 - z4,
            -x4 + y4 + z4, -x4 + y4 - z4, -x4 - y4 + z4, -x4 - y4 - z4
        ])

    return torch.stack(features, dim=1)

# 带残差连接的自注意力模块
class MultiAttentionWithResidual(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiAttentionWithResidual, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads)
        # 添加 LayerNorm 以增强稳定性
        self.norm = nn.BatchNorm1d(embed_dim)  

    def forward(self, x,y,z):
        # x shape: (B, N, C) -> (N, B, C) for multihead attention
        # 转换形状为 (N, B, C)
        x_permuted = x.permute(2, 0, 1)  
        y_permuted = y.permute(2, 0, 1)  
        z_permuted = z.permute(2, 0, 1)  
        
        # 自注意力和残差连接
        attn_output, _ = self.self_attention(x_permuted, y_permuted, z_permuted)
        
        # 转换回原始形状 (B, C, N)
        attn_output = attn_output.permute(1, 2, 0)
        
        # 添加残差连接和规范化
        x = self.norm(x + attn_output)
        
        return x

class PointNetSetAbstractionMsgSelectChoose(nn.Module):
    def __init__(self, npoint, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsgSelectChoose, self).__init__()
        self.npoint = npoint
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
        # 采样点
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i in range(len(self.nsample_list)):
            K = self.nsample_list[i]
            # 查询指定范围内的索引
            # group_idx = query_ball_point(radius, K, xyz, new_xyz)
            group_idx = nearest_point_stack(K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_points = index_points(points, group_idx)
            grouped_xyz_selected, grouped_points_selected = select_topk_by_cosine_similarity(grouped_xyz, grouped_points,round(2/3 * K), new_xyz)

            # 将邻域点坐标中心化，减去对应的采样点坐标 new_xyz
            # grouped_xyz_selected -= new_xyz.view(B, S, 1, C)
            grouped_points = torch.cat([grouped_points_selected, grouped_xyz_selected], dim=-1)
            # if points is not None:
            #     # 通过索引 group_idx 从 points 中提取邻域点的特征信息
            #     grouped_points = index_points(points, group_idx)
            #     # 将邻域点的特征与坐标信息拼接在一起，形状为 [B, S, K, D+C]
            #     grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            # else:
            #     grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        # 将 new_points_list 中的所有特征拼接在一起，形成最终的特征 new_points_concat，形状为 [B, D', S]
        new_points_concat = torch.cat(new_points_list, dim=1)
        # 返回采样后的点云坐标 和对应的特征 
        return new_xyz, new_points_concat

class PointNetSetAbstractionMsgSelectChooseUp(nn.Module):
    def __init__(self, npoint, nsample_list, in_channel, mlp_list,upper):
        super(PointNetSetAbstractionMsgSelectChooseUp, self).__init__()
        self.npoint = npoint
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        concat_channel = sum([sublist[-1] for sublist in mlp_list])
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
        # 升维
        self.conv_concat = nn.Conv1d(concat_channel,upper,1)
        # 降维
        self.conv_down = nn.Conv1d(upper,concat_channel,1)
        self.bn = nn.BatchNorm1d(concat_channel)
        self.attention = SelfAttentionWithResidual(concat_channel,8)
        

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
        # 采样点
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i in range(len(self.nsample_list)):
            K = self.nsample_list[i]
            # 查询指定范围内的索引
            # group_idx = query_ball_point(radius, K, xyz, new_xyz)
            group_idx = nearest_point_stack(K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_points = index_points(points, group_idx)
            grouped_xyz_selected, grouped_points_selected = select_topk_by_cosine_similarity(grouped_xyz, grouped_points,round(2/3 * K), new_xyz)

            # 将邻域点坐标中心化，减去对应的采样点坐标 new_xyz
            # grouped_xyz_selected -= new_xyz.view(B, S, 1, C)
            grouped_points = torch.cat([grouped_points_selected, grouped_xyz_selected], dim=-1)
            # if points is not None:
            #     # 通过索引 group_idx 从 points 中提取邻域点的特征信息
            #     grouped_points = index_points(points, group_idx)
            #     # 将邻域点的特征与坐标信息拼接在一起，形状为 [B, S, K, D+C]
            #     grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            # else:
            #     grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]

            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        # 将 new_points_list 中的所有特征拼接在一起，形成最终的特征 new_points_concat，形状为 [B, D', S]
        new_points_concat = torch.cat(new_points_list, dim=1)
        # 升维后降维
        # new_points_concat_up = F.relu(self.conv_concat(new_points_concat)) 
        # new_points_concat_up = self.conv_down(new_points_concat_up)
        # new_points_concat = self.bn(new_points_concat + new_points_concat_up)
        new_points_concat = self.attention(new_points_concat)

        # 返回采样后的点云坐标 和对应的特征 
        return new_xyz, new_points_concat

class PointNetSetAbstractionMsgSelectChooseAttention(nn.Module):
    def __init__(self, npoint, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsgSelectChooseAttention, self).__init__()
        self.npoint = npoint
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
        self.attention = SelfAttentionWithResidual(sum([sublist[-1] for sublist in mlp_list]),8)

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
        # 采样点
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i in range(len(self.nsample_list)):
            K = self.nsample_list[i]
            # 查询指定范围内的索引
            # group_idx = query_ball_point(radius, K, xyz, new_xyz)
            group_idx = nearest_point_stack(K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_points = index_points(points, group_idx)
            grouped_xyz_selected, grouped_points_selected = select_topk_by_cosine_similarity(grouped_xyz, grouped_points,round(2/3 * K), new_xyz)

            # 将邻域点坐标中心化，减去对应的采样点坐标 new_xyz
            # grouped_xyz_selected -= new_xyz.view(B, S, 1, C)
            grouped_points = torch.cat([grouped_points_selected, grouped_xyz_selected], dim=-1)
            # if points is not None:
            #     # 通过索引 group_idx 从 points 中提取邻域点的特征信息
            #     grouped_points = index_points(points, group_idx)
            #     # 将邻域点的特征与坐标信息拼接在一起，形状为 [B, S, K, D+C]
            #     grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            # else:
            #     grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        # 将 new_points_list 中的所有特征拼接在一起，形成最终的特征 new_points_concat，形状为 [B, D', S]
        new_points_concat = torch.cat(new_points_list, dim=1)
        new_points_concat = self.attention(new_points_concat)
        # 返回采样后的点云坐标 和对应的特征 
        return new_xyz, new_points_concat

class PointNetSetAbstractionAttention(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstractionAttention, self).__init__()
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
        self.attention = SelfAttentionWithResidual(mlp[-1],8)


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
        for i, conv in enumerate(self.mlp_convs): # 暴力从515->256->512->1024
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))
        # 对第3维进行最大化处理
        new_points = torch.max(new_points, 2)[0]
        new_points = self.attention(new_points)
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

# 用于恢复分辨率
class PointNetFeaturePropagationAttention(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagationAttention, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
        self.attention = SelfAttentionWithResidual(mlp[-1],8)

    def forward(self, xyz1, xyz2, points1, points2):
        # 点云上采样
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
            # 如果采样点集有多个点 (S > 1)，则需要进行插值计算
            # 计算 xyz1 中每个点到 xyz2 中各个点的平方距离
            dists = square_distance(xyz1, xyz2)
            # 按距离升序排序
            dists, idx = dists.sort(dim=-1)
            # 取距离最近的三个点，进行三线性插值，保留前3个最小距离及其对应索引
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
            # 计算距离的倒数，以便生成权重，1e-8 是为了避免除零错误
            dist_recip = 1.0 / (dists + 1e-8)
            # 计算权重的归一化因子 norm，用于归一化每个点的权重
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            # 生成插值权重 weight，使其总和为1
            weight = dist_recip / norm
            # 根据插值权重对 points2 中的最近3个点的特征进行加权求和，得到插值后的特征 interpolated_points
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
        new_points = self.attention(new_points)
        return new_points

class PointNetSetAbstractionMsgAttention(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsgAttention, self).__init__()
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
        self.attention = SelfAttentionWithResidual(sum([sublist[-1] for sublist in mlp_list]),8)
        

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
        # 采样点
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            # 查询指定范围内的索引
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            # 将邻域点坐标中心化，减去对应的采样点坐标 new_xyz
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                # 通过索引 group_idx 从 points 中提取邻域点的特征信息
                grouped_points = index_points(points, group_idx)
                # 将邻域点的特征与坐标信息拼接在一起，形状为 [B, S, K, D+C]
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
        # 将 new_points_list 中的所有特征拼接在一起，形成最终的特征 new_points_concat，形状为 [B, D', S]
        new_points_concat = torch.cat(new_points_list, dim=1)
        new_points_concat = self.attention(new_points_concat)
        # 返回采样后的点云坐标 和对应的特征 
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

    # 计算采样点和邻居点的平方距离，得到距离矩阵 [B, S, K]
    sqrdists = torch.sum((grouped_points - new_xyz.unsqueeze(2)) ** 2, dim=-1)  # [B, S, K]

    group_idx = []
    
    for b in range(B):
        batch_group_idx = []
        for s in range(S):
            # 根据距离筛选出半径内的邻居点（距离小于 radius[b, s] 的点）
            valid_idx = torch.where(sqrdists[b, s] < radius[b, s] ** 2)[0]  # 获取满足条件的点索引
            if valid_idx.shape[0] == 0:  # 如果没有邻居点，确保至少保留第一个邻居
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
    # 计算所有邻居点到采样点的平方距离 [B, S, K]
    sqrdists = torch.sum((grouped_points - new_xyz.unsqueeze(2)) ** 2, dim=-1)  # [B, S, K]

    # 根据距离半径创建一个掩码 mask，标记哪些邻居在半径范围内 [B, S, K]
    mask = sqrdists < radius.unsqueeze(-1) ** 2  # [B, S, K]
    
    return mask  # 返回掩码，用于后续的邻居筛选

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

    # 计算每个邻居点到采样点的平方距离，得到距离矩阵 [B, S, K]
    sqrdists = torch.sum((grouped_points - new_xyz.unsqueeze(2)) ** 2, dim=-1)  # [B, S, K]

    # 创建一个掩码，标记哪些邻居点在半径内
    mask = sqrdists < radius.unsqueeze(-1) ** 2  # [B, S, K] 掩码为True表示在半径内的邻居点

    # 对每个采样点有效的邻居进行随机打乱并选择3/4的邻居
    valid_counts = mask.sum(dim=-1, keepdim=True)  # 计算每个采样点内有效邻居的数量 [B, S, 1]

    # 防止 division by zero 的问题，确保每个有效的邻居数大于1，至少采样一个邻居
    num_sampled = torch.clamp((ratio * valid_counts).int(), min=1)  # 计算保留的3/4的邻居数量 [B, S, 1]

    # 随机打乱邻居顺序并选择前num_sampled个邻居
    rand_idx = torch.argsort(torch.rand(B, S, K, device=device), dim=-1)  # 随机打乱每个采样点的邻居顺序 [B, S, K]
    sampled_idx = torch.arange(K, device=device).unsqueeze(0).unsqueeze(0) < num_sampled  # 生成mask用以选择3/4的邻居 [B, S, K]

    # 生成最终掩码，表示3/4随机采样的邻居
    mask = mask & sampled_idx.scatter(2, rand_idx, sampled_idx)  # 将随机采样的邻居选择为True

    return mask  # 返回经过随机采样后的掩码

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

    # 计算距离矩阵 [B, S, K]
    sqrdists = torch.sum((grouped_points - new_xyz.unsqueeze(2)) ** 2, dim=-1)  # [B, S, K]

    # 创建掩码：在半径范围内的点为 True，否则为 False
    mask = sqrdists < radius.unsqueeze(-1) ** 2  # [B, S, K]

    # 每个点的有效邻居数量 [B, S]
    valid_counts = mask.sum(dim=-1)  # [B, S]

    # 获取第一个有效邻居的索引 [B, S, 1]
    first_valid_idx = mask.float().argmax(dim=-1, keepdim=True)  # [B, S, 1]

    # 扩展第一个有效邻居索引，以适应最终的 k 个邻居点 [B, S, k]
    first_valid_idx = first_valid_idx.repeat(1, 1, k)  # [B, S, k]

    # 构建初始邻居索引 [B, S, K]
    selected_idx = torch.arange(K, device=grouped_points.device).view(1, 1, K).repeat(B, S, 1)  # [B, S, K]

    # 确保掩码的形状与邻居索引匹配 [B, S, K]
    mask = mask.unsqueeze(-1).expand(-1, -1, K)  # [B, S, K]

    # 使用掩码选择有效邻居索引，否则填充为第一个有效邻居索引 [B, S, k]
    padded_idx = torch.where(mask[:, :, :k], selected_idx[:, :, :k], first_valid_idx)  # [B, S, k]

    return padded_idx  # [B, S, k]

# 对拼接后的维度，先升高，后降低
class PointNetSetAbstractionMsgSelectPyramid(nn.Module):
    def __init__(self, npoint, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsgSelectPyramid, self).__init__()
        self.npoint = npoint
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
        channel_out = sum([sublist[-1] for sublist in mlp_list])
        self.attention = SelfAttentionWithResidual(2 * channel_out,8)
        self.conv_down = nn.Conv1d(2 * channel_out,channel_out,1)
        self.bn_down = nn.BatchNorm1d(channel_out)
        self.act = nn.Softplus()   


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
        # 采样点
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i in range(len(self.nsample_list)):
            K = self.nsample_list[i]
            # 查询指定范围内的索引
            # group_idx = query_ball_point(radius, K, xyz, new_xyz)
            group_idx = nearest_point_stack(K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_points = index_points(points, group_idx)
            grouped_xyz_selected, grouped_points_selected = select_topk_by_cosine_similarity(grouped_xyz, grouped_points,round(2/3 * K), new_xyz)

            # 将邻域点坐标中心化，减去对应的采样点坐标 new_xyz
            grouped_xyz_selected -= new_xyz.view(B, S, 1, C)
            grouped_points = torch.cat([grouped_points_selected, grouped_xyz_selected], dim=-1)
            # if points is not None:
            #     # 通过索引 group_idx 从 points 中提取邻域点的特征信息
            #     grouped_points = index_points(points, group_idx)
            #     # 将邻域点的特征与坐标信息拼接在一起，形状为 [B, S, K, D+C]
            #     grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            # else:
            #     grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        # 将 new_points_list 中的所有特征拼接在一起，形成最终的特征 new_points_concat，形状为 [B, D', S]
        # 维度翻倍
        new_points_concat = torch.cat(new_points_list, dim=1)
        new_points_concat = torch.cat([new_points_concat,new_points_concat],dim=1)
        new_points_concat = self.attention(new_points_concat)
        # 然后降维
        new_points_concat = self.act(self.bn_down(self.conv_down(new_points_concat)))
        # 返回采样后的点云坐标 和对应的特征 
        return new_xyz, new_points_concat

def knn_point(k, xyz, new_xyz):
    """
    使用 KNN 找到每个采样点的最近 k 个邻居。
    Input:
        k: 每个采样点的最近邻个数
        xyz: 所有点云 [B, N, 3]
        new_xyz: 采样点 [B, S, 3]
    Return:
        group_idx: 每个采样点的邻居索引 [B, S, k]
        grouped_xyz: 这些邻居点的坐标 [B, S, k, 3]
    """
    sqrdists = square_distance(new_xyz, xyz)  # [B, S, N]
    _, group_idx = torch.topk(sqrdists, k, dim=-1, largest=False, sorted=True)
    
    # 根据索引获取邻居点的坐标
    grouped_xyz = index_points(xyz, group_idx)  # [B, S, k, 3]
    return group_idx, grouped_xyz

def local_query_ball(radius, nsample, grouped_xyz):
    """
    在局部邻域中进行球状采样。
    Input:
        radius: 球半径
        nsample: 每个球中最多采样的邻居数量
        grouped_xyz: 局部邻域中的邻居点坐标 [B, S, k, 3]
    Return:
        group_idx: 每个采样点的邻居索引 [B, S, nsample]
    """
    B, S, k, _ = grouped_xyz.shape

    # 计算每对点之间的平方距离 [B, S, k]
    center_xyz = grouped_xyz[:, :, 0:1, :]  # 取第一个邻居作为中心 [B, S, 1, 3]
    dists = torch.sum((grouped_xyz - center_xyz) ** 2, dim=-1)  # [B, S, k]

    # 找出在 radius 范围内的点，并按距离排序
    mask = dists <= radius ** 2
    group_idx = torch.arange(k, dtype=torch.long, device=grouped_xyz.device).view(1, 1, k).repeat(B, S, 1)
    group_idx[~mask] = k  # 超过半径的点用 k 进行标记

    # 对有效点按距离排序，并截取前 nsample 个邻居
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    # 不够 nsample 个时用第一个点补齐
    group_first = group_idx[:, :, 0].unsqueeze(-1).repeat(1, 1, nsample)
    mask = group_idx == k
    group_idx[mask] = group_first[mask]

    return group_idx

# 用KNN寻找采样点附近的邻居，根据邻居最小距离和最大距离，分为3个尺度，分别收集邻居信息
class PointNetSetAbstractionMsgKnnDistance(nn.Module):
    def __init__(self, npoint, K, in_channel, mlp_list):
        super(PointNetSetAbstractionMsgKnnDistance, self).__init__()
        self.npoint = npoint
        self.K = K
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
            xyz: input points position data, [B, 3, N] (坐标信息)
            points: input points feature data, [B, D, N] (特征信息)
        Return:
            new_xyz: sampled points position data, [B, 3, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)  # [B, N, 3]，将坐标信息转为 [B, N, 3]
        points = points.permute(0, 2, 1)  # [B, N, D]，将特征信息转为 [B, N, D]

        B, N, C = xyz.shape
        S = self.npoint  # 采样点的数量

        # 使用最远点采样 (farthest point sampling) 获取采样点
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))  # [B, S, 3]
        
        # -----
        group_idx, grouped_xyz = knn_point(self.K, xyz, new_xyz)
        radius_scales = [1/3, 2/3, 1.0]
        new_points_list = []
        # 对每种比例的球状采样进行处理
        for i,scale in enumerate(radius_scales):
            # 动态计算每个采样点的半径范围：p + scale * (q - p)
            dists = torch.norm(grouped_xyz - grouped_xyz[:, :, 0:1, :], dim=-1)  # [B, S, k]
            p = dists.min(dim=-1, keepdim=True)[0]  # 最近邻距离 [B, S, 1]
            q = dists.max(dim=-1, keepdim=True)[0]  # 最远邻距离 [B, S, 1]
            radius = p + scale * (q - p)  # 动态半径 [B, S, 1]

            # 扩展维度与邻居数量对齐
            radius = radius.expand(-1, -1, self.K)

            # 在局部邻域中进行球状采样
            batch_group_idx = local_query_ball(radius, self.K, grouped_xyz)
            group_points = index_points(points,batch_group_idx)
            grouped_xyz_new = index_points(xyz,batch_group_idx)
            grouped_xyz_new -= new_xyz.view(B, S, 1, C)
            group_points = torch.cat([group_points,grouped_xyz_new],dim=-1)
            group_points = group_points.permute(0, 3, 2, 1)  # [B, D, K, S]

            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                group_points = F.relu(bn(conv(group_points)))  # [B, D', selected_K, S]

            # 最大池化聚合邻居特征
            new_points = torch.max(group_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)
            # 将结果添加到批次列表中
            # group_indices_batches.append(batch_group_idx)

        # -----



        # # 使用 KNN 获取采样点的 K 个最近邻居
        # neighbor_idx = knn(xyz, new_xyz, self.K)  # [B, S, K]

        # # 获取K个最近邻居，并计算尺度
        # grouped_points = index_points(xyz, neighbor_idx)  # [B, S, K, 3]
        # scales = compute_scales(grouped_points)  # [B, S, 3]
        
        # # Step 4: Query ball points at three different scales using masks and extract features
        # new_points_list = []  # 用于存储不同尺度下的邻居特征
        # for i in range(3):  # for each scale (s1, s2, s3)
        #     radius = scales[:, :, i]  # [B, S] - radius for the current scale
        #     mask = query_ball_point_new(radius, grouped_points, new_xyz)  # [B, S, K] - mask indicating valid neighbors

        #     # 使用 mask 从 points 中提取邻居的特征
        #     expanded_points = index_points(points, neighbor_idx)  # [B, S, K, D]
        #     expanded_xyz = index_points(xyz, neighbor_idx)  # [B, S, K, 3]

        #     # 使用 mask 选择有效的邻居特征
        #     valid_neighbors = expanded_points * mask.unsqueeze(-1).float()  # [B, S, K, D] - apply mask to features
        #     valid_neighbors_xyz = expanded_xyz * mask.unsqueeze(-1).float()  # [B, S, K, 3] - apply mask to coordinates

        #     selected_points = torch.cat([valid_neighbors,valid_neighbors_xyz],dim=-1)
        #     selected_points = selected_points.permute(0, 3, 2, 1)  # [B, D, K, S]
        #     for j in range(len(self.conv_blocks[i])):
        #         conv = self.conv_blocks[i][j]
        #         bn = self.bn_blocks[i][j]
        #         selected_points = F.relu(bn(conv(selected_points)))  # [B, D', selected_K, S]

        #     # 最大池化聚合邻居特征
        #     new_points = torch.max(selected_points, 2)[0]  # [B, D', S]
        #     new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)  # 恢复为 [B, 3, S]
        new_points_concat = torch.cat(new_points_list, dim=1)  # 将特征拼接 [B, D', S]

        return new_xyz, new_points_concat

class PointNetSetAbstractionMsgSelectV2(nn.Module):
    def __init__(self, npoint, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsgSelectV2, self).__init__()
        self.npoint = npoint
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel
            for j in range(len(mlp_list[i])):
                if j != 0:
                    last_channel = out_channel + 3
                out_channel = mlp_list[i][j]
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
        # 采样点
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i in range(len(self.nsample_list)):
            K = self.nsample_list[i]
            # 查询指定范围内的索引
            group_idx = nearest_point_stack(K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_points = index_points(points, group_idx)
            grouped_xyz_selected, grouped_points_selected = select_topk_by_cosine_similarity(grouped_xyz, grouped_points,round(3/4 * K), new_xyz)
            # 将邻域点坐标中心化，减去对应的采样点坐标 new_xyz
            grouped_xyz_selected -= new_xyz.view(B, S, 1, C)
            grouped_points_selected = grouped_points_selected.permute(0, 3, 2, 1)  # [B, D, K, S]
            grouped_xyz_selected = grouped_xyz_selected.permute(0, 3, 2, 1)
            # 对grouped_points_selected先进行1次MLP
            # 对拼接后的再进行2次MLP
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                if j != 0:
                    # 非第一次，采用拼接
                    grouped_points_selected = torch.cat([grouped_points_selected, grouped_xyz_selected], dim=1)
                grouped_points_selected =  F.relu(bn(conv(grouped_points_selected)))

            grouped_points_selected = torch.max(grouped_points_selected, 2)[0]  # [B, D', S]
            new_points_list.append(grouped_points_selected)

        new_xyz = new_xyz.permute(0, 2, 1)
        # 将 new_points_list 中的所有特征拼接在一起，形成最终的特征 new_points_concat，形状为 [B, D', S]
        new_points_concat = torch.cat(new_points_list, dim=1)
        # 返回采样后的点云坐标 和对应的特征 
        return new_xyz, new_points_concat

class PointNetSetAbstractionMsgSelectV3(nn.Module):
    def __init__(self, npoint, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsgSelectV3, self).__init__()
        self.npoint = npoint
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
        channel_out = sum([sublist[-1] for sublist in mlp_list])
        self.up = nn.Sequential(nn.Conv1d(channel_out,2 * channel_out,1),
                                nn.BatchNorm1d(2 * channel_out),
                                nn.ReLU())
        self.down = nn.Sequential(nn.Conv1d(2 *channel_out, channel_out,1),
                                nn.BatchNorm1d(channel_out),
                                nn.ReLU())

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
        # 采样点
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i in range(len(self.nsample_list)):
            K = self.nsample_list[i]
            # 查询指定范围内的索引
            # group_idx = query_ball_point(radius, K, xyz, new_xyz)
            group_idx = nearest_point_stack(K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_points = index_points(points, group_idx)
            grouped_xyz_selected, grouped_points_selected = select_topk_by_cosine_similarity(grouped_xyz, grouped_points,round(3/4 * K), new_xyz)

            # 将邻域点坐标中心化，减去对应的采样点坐标 new_xyz
            grouped_xyz_selected -= new_xyz.view(B, S, 1, C)
            grouped_points = torch.cat([grouped_points_selected, grouped_xyz_selected], dim=-1)

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        # 将 new_points_list 中的所有特征拼接在一起，形成最终的特征 new_points_concat，形状为 [B, D', S]
        new_points_concat = torch.cat(new_points_list, dim=1)
        # 对整体特征先升维，再降维，模拟倒瓶颈结构
        new_points_concat = self.down(self.up(new_points_concat))
        # 返回采样后的点云坐标 和对应的特征 
        return new_xyz, new_points_concat

class PointNetSetAbstractionV4(nn.Module):
    def __init__(self,  in_channel, mlp):
        super(PointNetSetAbstractionV4, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
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
        B,C,N = points.shape
        # # xyz = xyz.permute(0, 2, 1)
        # if points is not None:
        #     points = points.permute(0, 2, 1)

        # B 3 512 512 
        # xyz_extend = xyz.unsqueeze(-1).expand(-1, -1, -1, self.npoint // 4 )
        # B 320 512 512
        # points_extend = points.unsqueeze(-1).expand(-1, -1, -1, self.npoint // 4)
        # B 323 512 512
        new_points = torch.cat([points,xyz],dim=1)

        # 
        # new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs): # 暴力从515->256->512->1024
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))
        # 对第3维进行最大化处理
        # new_points = torch.max(new_points, 3)[0]
        return  new_points

# 用于恢复分辨率
# 原理：距离越近的点，特征也是相似的
class PointNetFeaturePropagationV85(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagationV85, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        # 点云上采样
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
            # 如果采样点集有多个点 (S > 1)，则需要进行插值计算
            # 计算 xyz1 中每个点到 xyz2 中各个点的平方距离
            dists = square_distance(xyz1, xyz2)
            # 按距离升序排序
            dists, idx = dists.sort(dim=-1)
            # 取距离最近的三个点，进行三线性插值，保留前3个最小距离及其对应索引
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
            # 计算距离的倒数，以便生成权重，1e-8 是为了避免除零错误
            dist_recip = 1.0 / (dists + 1e-8)
            # 计算权重的归一化因子 norm，用于归一化每个点的权重
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            # 生成插值权重 weight，使其总和为1
            weight = dist_recip / norm
            # 根据插值权重对 points2 中的最近3个点的特征进行加权求和，得到插值后的特征 interpolated_points
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points,xyz1], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

class PointNetSetAbstractionMsgSelectNormalV11(nn.Module):
    def __init__(self, npoint, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsgSelectNormalV11, self).__init__()
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
        # 采样点
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        weights = torch.ones(B, N).to(xyz.device)
        for i in range(len(self.nsample_list)):
            K = self.nsample_list[i]
            # 查询指定范围内的索引
            # group_idx = query_ball_point(radius, K, xyz, new_xyz)
            group_idx = nearest_point_stack(K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_points = index_points(points, group_idx)
            neighborhood_weights = torch.gather(weights.unsqueeze(1).expand(-1, S, -1), 2, group_idx)  # [B, S, K]
            sampled_weights = neighborhood_weights.sum(dim=-1)  # [B, S]
            
            # 更新采样点的权重，将新权重赋回到相应位置
            weights = weights.scatter(1, group_idx.view(B, -1), sampled_weights.unsqueeze(-1).expand(-1, -1, K).reshape(B, -1))
            grouped_xyz_selected, grouped_points_selected,selected_normals = select_topk_by_cosine_similarity_normal(grouped_xyz, grouped_points,round(3/4 * K), new_xyz)


            # 将邻域点坐标中心化，减去对应的采样点坐标 new_xyz
            grouped_xyz_selected -= new_xyz.view(B, S, 1, C)
            # 拼接上法线信息
            grouped_points = torch.cat([grouped_points_selected, grouped_xyz_selected,selected_normals], dim=-1)

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        # 将 new_points_list 中的所有特征拼接在一起，形成最终的特征 new_points_concat，形状为 [B, D', S]
        new_points_concat = torch.cat(new_points_list, dim=1)
        # 返回采样后的点云坐标 和对应的特征 
        return new_xyz, new_points_concat,weights

# 返回法线信息，增加xyz的特征
def select_topk_by_cosine_similarity_normalV11(grouped_xyz, grouped_points, k, new_xyz):
    """
    根据余弦相似度选择邻域中的前k个点，并返回其法线。
    Args:
        grouped_xyz: 邻域点的坐标，形状为 [B, S, K, C]
        grouped_points: 邻域点的特征，形状为 [B, S, K, D]
        k: 需要选择的点的数量
        new_xyz: 采样点坐标，形状为 [B, S, C]
    Returns:
        selected_grouped_xyz: 选择后的坐标，形状为 [B, S, k, C]
        selected_grouped_points: 选择后的特征，形状为 [B, S, k, D]
        selected_normals: 选择后的法线，形状为 [B, S, k, C]
    """
    # 计算邻域点和采样点的法线
    normals_grouped = compute_normals(grouped_xyz)  # [B, S, K, C]
    normals_new = compute_normals(new_xyz.unsqueeze(2))  # [B, S, 1, C]
    
    # 计算余弦相似度
    cosine_similarity = compute_cosine_similarity(normals_grouped, normals_new)  # [B, S, K]
    
    # 对相似度排序，选择前k个
    topk_weights,topk_indices = torch.topk(cosine_similarity, k, dim=-1)  # [B, S, k]
    # 取均值表示总权重
    neighborhood_weights = torch.mean(topk_weights, dim=-1)
    # 根据索引选择坐标和特征
    # 使用`torch.gather`选择坐标
    selected_grouped_xyz = torch.gather(
        grouped_xyz,  # [B, S, K, C]
        2,
        topk_indices.unsqueeze(-1).expand(-1, -1, -1, grouped_xyz.shape[-1])  # [B, S, k, C]
    )

    # 选择对应的法线
    selected_normals = torch.gather(
        normals_grouped,  # [B, S, K, C]
        2,
        topk_indices.unsqueeze(-1).expand(-1, -1, -1, normals_grouped.shape[-1])  # [B, S, k, C]
    )

    # 如果还需要选择特征
    selected_grouped_points = torch.gather(
        grouped_points,  # [B, S, K, D]
        2,
        topk_indices.unsqueeze(-1).expand(-1, -1, -1, grouped_points.shape[-1])  # [B, S, k, D]
    )

    return selected_grouped_xyz, selected_grouped_points, selected_normals,neighborhood_weights

# 用于恢复分辨率
# 原理：距离越近的点，特征也是相似的
class PointNetFeaturePropagationV11(nn.Module):
    def __init__(self, in_channel, mlp,channel,attention=True):
        super(PointNetFeaturePropagationV11, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
        self.bn1 = nn.BatchNorm1d(channel)
        self.att = attention
        self.attention = SelfAttentionWithResidual(embed_dim=channel,num_heads=8)


    def forward(self, xyz1, xyz2, points1, points2,weights=None):
        # 点云上采样
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
            # 如果采样点集有多个点 (S > 1)，则需要进行插值计算
            # 计算 xyz1 中每个点到 xyz2 中各个点的平方距离
            dists = square_distance(xyz1, xyz2)
            # 按距离升序排序
            dists, idx = dists.sort(dim=-1)
            # 取距离最近的三个点，进行三线性插值，保留前3个最小距离及其对应索引
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
            # 计算距离的倒数，以便生成权重，1e-8 是为了避免除零错误
            dist_recip = 1.0 / (dists + 1e-8)
            # 计算权重的归一化因子 norm，用于归一化每个点的权重
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            # 生成插值权重 weight，使其总和为1
            # 拼接上
            weight = dist_recip / norm
            # 根据插值权重对 points2 中的最近3个点的特征进行加权求和，得到插值后的特征 interpolated_points
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
            # weights = weights.view(B,N,1).repeat(1,1,S)
            # dis_points = torch.matmul(weights,points2)
            # weights = weights.view(B,N,1)
            # dis_points = torch.matmul(points1,weights).permute(0,2,1).repeat(1,N,1)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            # interpolated_points = self.bn1(interpolated_points)
            if self.att is True:
                interpolated_points = interpolated_points.permute(0,2,1)
                interpolated_points = self.attention(interpolated_points)
                interpolated_points = interpolated_points.permute(0,2,1)
            new_points = torch.cat([points1,interpolated_points], dim=-1)
            # new_points = torch.cat([interpolated_points, dis_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

def feature_distance(points1, points2):
    """
    Calculate feature distance between each point in points1 and each point in points2.

    Input:
        points1: source points features, shape [B, N, D1]
        points2: target points features, shape [B, M, D2]
    Output:
        dist: per-point feature distance, shape [B, N, M]
    """
    B, N, D1 = points1.shape
    _, M, D2 = points2.shape

    # 如果 D1 和 D2 不一致，则将较低维度的特征映射到较高维度
    if D1 > D2:
        points2 = F.linear(points2, torch.randn(D1, D2).to(points2.device))  # 将 points2 映射到 D1 维度
    elif D2 > D1:
        points1 = F.linear(points1, torch.randn(D2, D1).to(points1.device))  # 将 points1 映射到 D2 维度

    # 更新特征维度，使得两者相同
    D = max(D1, D2)

    # 扩展 points1 和 points2 的维度以便进行广播
    points1_expanded = points1.unsqueeze(2).expand(-1, -1, M, -1)  # [B, N, M, D]
    points2_expanded = points2.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, M, D]

    # 计算每对点特征之间的欧氏距离平方和
    dist = torch.sum((points1_expanded - points2_expanded) ** 2, dim=-1)  # [B, N, M]

    return dist

# 用于恢复分辨率
# 原理：距离越近的点，特征也是相似的
class PointNetFeaturePropagationV12(nn.Module):
    def __init__(self, in_channel, mlp,use_feature=False,num=0):
        super(PointNetFeaturePropagationV12, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
        self.bn = nn.BatchNorm1d(num)
        self.use_feature = use_feature
        self.wn = WeightNetV12(2 * num,num)

    def forward(self, xyz1, xyz2, points1, points2):
        # 点云上采样
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
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            # 如果采样点集有多个点 (S > 1)，则需要进行插值计算
            # 计算 xyz1 中每个点到 xyz2 中各个点的平方距离
            space_dists = square_distance(xyz1, xyz2)
            if self.use_feature is True:
                feature_dists = feature_distance(points1,points2)
                combined_dists = torch.cat([space_dists, feature_dists], dim=-1)
                weights = self.wn(combined_dists)
                interpolated_points = torch.matmul(weights,points2)
            else:
                dists = space_dists
                # 按距离升序排序
                dists, idx = dists.sort(dim=-1)
                # 取距离最近的三个点，进行三线性插值，保留前3个最小距离及其对应索引
                dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
                # 计算距离的倒数，以便生成权重，1e-8 是为了避免除零错误
                dist_recip = 1.0 / (dists + 1e-8)
                # 计算权重的归一化因子 norm，用于归一化每个点的权重
                norm = torch.sum(dist_recip, dim=2, keepdim=True)
                # 生成插值权重 weight，使其总和为1
                weight = dist_recip / norm
                # 根据插值权重对 points2 中的最近3个点的特征进行加权求和，得到插值后的特征 interpolated_points
                interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        new_points = torch.cat([points1, interpolated_points], dim=-1)

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

class WeightNetV12(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(WeightNetV12, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels)
        self.fc2 = nn.Linear(in_channels, out_channels)
    
    def forward(self, dists):
        x = F.relu(self.fc1(dists))
        weights = F.softmax(self.fc2(x), dim=-1)
        return weights

# 线性自注意力
# 通过减少点积注意力的计算复杂度来减少显存需求
class LinearSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(LinearSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # 使用低秩近似的线性映射
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # 输出层
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.BatchNorm1d(embed_dim)  # 增强稳定性

    def forward(self, x):
        # x shape: (B, C, N)
        B, C, N = x.shape
        x = x.permute(0,2,1)  # 转换为 (B,N, C)
        
        # 计算 query, key, value
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)
        
        # 使用线性注意力近似计算
        # 将 attention map 的计算转化为矩阵乘法的形式，减少复杂度
        attn_weights = torch.einsum("nbe,nbe->be", Q, K) / self.embed_dim ** 0.5
        attn_output = torch.einsum("be,nbe->nbe", attn_weights, V)
        
        # 转换回原始形状 (B, N, C) 并应用残差连接
        attn_output = attn_output.permute(0,2,1)  # (B,N, C) -> (B, C, N)
        
        # 添加残差和规范化
        x = self.norm(x.permute(0,2,1) + attn_output)
        
        return x

# 稀疏自注意力
# 稀疏注意力通过仅计算局部窗口内的注意力分数，忽略长距离依赖关系来减少计算量。适合大规模点云的情况
# 稀疏注意力适用于局部相关性强的点云数据
class SparseSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=4):
        super(SparseSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        # x shape: (B, C, N)
        B, C, N = x.shape
        
        x = x.permute(2, 0, 1)  # 转换为 (N, B, C)
        
        # 将 N 维度划分为窗口
        windows = x.unfold(dimension=0, size=self.window_size, step=self.window_size)
        windows = windows.contiguous().view(-1, self.window_size, C).permute(1, 0, 2)  # (window_size, B * num_windows, C)
        
        attn_output, _ = self.self_attention(windows, windows, windows)
        
        attn_output = attn_output.permute(1, 2, 0).contiguous().view(B, C, -1)  # 转换回 (B, C, N)
        
        x = self.norm(x.permute(1, 2, 0) + attn_output)
        
        return x

# 因子化自注意力
# 因子化注意力将原始注意力分解为较小的矩阵乘法，使得复杂度降低。适用于长序列的情况
# 因子化注意力适用于长序列数据，适合多头注意力场景下的效率提升
class FactorizedSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, factor=4):
        super(FactorizedSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.factor = factor
        # 投影层，将Q和K映射到降维空间
        self.query_proj = nn.Linear(embed_dim, embed_dim // factor)
        self.key_proj = nn.Linear(embed_dim, embed_dim // factor)
        # 将V也映射到相同的降维空间
        self.value_proj = nn.Linear(embed_dim, embed_dim // factor)
        # 将输出投影回嵌入维度
        self.out_proj = nn.Linear(embed_dim // factor, embed_dim)
        self.norm = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        # x shape: (B, C, N)
        B, C, N = x.shape
        x = x.permute(2, 0, 1)  # 转换为 (N, B, C)
        
        # 计算Q, K, V
        Q = self.query_proj(x)  # (N, B, C // factor)
        K = self.key_proj(x)    # (N, B, C // factor)
        V = self.value_proj(x)  # (N, B, C // factor)
        
        # 计算注意力权重
        attn_weights = torch.einsum("nbe,nbe->be", Q, K) / (self.embed_dim // self.factor) ** 0.5
        # 将注意力权重应用到V
        attn_output = torch.einsum("be,nbe->nbe", attn_weights, V)
        
        # 将attn_output投影回原始嵌入维度
        attn_output = self.out_proj(attn_output)  # (N, B, C)
        
        # 转换回原始形状 (B, C, N)
        attn_output = attn_output.permute(1, 2, 0)
        
        # 残差连接和规范化
        x = self.norm(x.permute(1, 2, 0) + attn_output)
        
        return x

class PointNetFeaturePropagationV13(nn.Module):
    def __init__(self, in_channel, mlp,num):
        super(PointNetFeaturePropagationV13, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
        self.bn =nn.BatchNorm1d(num)

    def forward(self, xyz1, xyz2, points1, points2):
        # 点云上采样
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
        _,D1,N1 = points1.shape
        _,D2,N2 = points2.shape
        points2 = points2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            # 如果采样点集有多个点 (S > 1)，则需要进行插值计算
            # 计算 xyz1 中每个点到 xyz2 中各个点的平方距离
            space_dists = square_distance(xyz1, xyz2)
            # 对少的特征进行0填充
            padding_size = D2 - D1
            points1_expanded = F.pad(points1, (0,  padding_size), "constant", 0)
            points_dists = square_distance(points1_expanded,points2)
            dists = self.bn((space_dists + points_dists).permute(0,2,1)).permute(0,2,1)
            # 按距离升序排序
            dists, idx = dists.sort(dim=-1)
            # 取距离最近的三个点，进行三线性插值，保留前3个最小距离及其对应索引
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
            # 计算距离的倒数，以便生成权重，1e-8 是为了避免除零错误
            dist_recip = 1.0 / (dists + 1e-8)
            # 计算权重的归一化因子 norm，用于归一化每个点的权重
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            # 生成插值权重 weight，使其总和为1
            weight = dist_recip / norm
            # 根据插值权重对 points2 中的最近3个点的特征进行加权求和，得到插值后的特征 interpolated_points
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points
# 显存需求过高
class GraphBasedFactorizedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, factor=4, k_neighbors=16):
        super(GraphBasedFactorizedAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.factor = factor
        self.k_neighbors = k_neighbors

        self.query_proj = nn.Linear(embed_dim, embed_dim // factor)
        self.key_proj = nn.Linear(embed_dim, embed_dim // factor)
        self.value_proj = nn.Linear(embed_dim, embed_dim // factor)
        self.out_proj = nn.Linear(embed_dim // factor, embed_dim)
        self.norm = nn.BatchNorm1d(embed_dim)
    
    def build_knn_graph(self, x, k=16):
        B, C, N = x.shape
        x = x.permute(0, 2, 1)
        dist = torch.cdist(x, x)
        knn_idx = dist.topk(k, largest=False, dim=-1).indices
        return knn_idx

    def forward(self, x):
        B, C, N = x.shape
        knn_idx = self.build_knn_graph(x, self.k_neighbors)
        x = x.permute(2, 0, 1)

        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)

        # 基于图的邻居点索引计算注意力
        attn_weights = torch.einsum("nbe,nbe->be", Q[knn_idx], K[knn_idx]) / (self.embed_dim // self.factor) ** 0.5
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        attn_output = torch.einsum("be,nbe->nbe", attn_weights, V[knn_idx])
        attn_output = self.out_proj(attn_output)
        attn_output = attn_output.permute(1, 2, 0)
        
        x = self.norm(x.permute(1, 2, 0) + attn_output)
        
        return x
# 显存需求过高
class PointwiseFactorizedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, factor=4, radius=0.5):
        super(PointwiseFactorizedAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.factor = factor
        self.radius = radius  # 限制注意力计算的半径范围

        self.query_proj = nn.Linear(embed_dim, embed_dim // factor)
        self.key_proj = nn.Linear(embed_dim, embed_dim // factor)
        self.value_proj = nn.Linear(embed_dim, embed_dim // factor)
        self.out_proj = nn.Linear(embed_dim // factor, embed_dim)
        self.norm = nn.BatchNorm1d(embed_dim)

    def get_radius_neighbors(self, x, radius):
        B, C, N = x.shape
        x = x.permute(0, 2, 1)  # (B, N, C)
        dist = torch.cdist(x, x)  # (B, N, N)
        neighbors = dist < radius  # (B, N, N) 选取半径范围内的邻居
        return neighbors

    def forward(self, x):
        B, C, N = x.shape
        neighbors = self.get_radius_neighbors(x, self.radius)  # 获取每个点的半径范围邻居
        x = x.permute(2, 0, 1)  # 转为 (N, B, C)

        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)
        
        # 仅在邻域内计算注意力
        attn_weights = torch.einsum("nbe,nbe->be", Q[neighbors], K[neighbors]) / (self.embed_dim // self.factor) ** 0.5
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        attn_output = torch.einsum("be,nbe->nbe", attn_weights, V[neighbors])
        attn_output = self.out_proj(attn_output)  
        attn_output = attn_output.permute(1, 2, 0)

        x = self.norm(x.permute(1, 2, 0) + attn_output)
        
        return x
# 显存需求过高
class LocalBlockAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, factor=4, k_neighbors=16):
        super(LocalBlockAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.factor = factor
        self.k_neighbors = k_neighbors  # 每个点的邻居数量
        
        # 投影层，将Q, K, V映射到降维空间
        self.query_proj = nn.Linear(embed_dim, embed_dim // factor)
        self.key_proj = nn.Linear(embed_dim, embed_dim // factor)
        self.value_proj = nn.Linear(embed_dim, embed_dim // factor)
        # 将输出投影回嵌入维度
        self.out_proj = nn.Linear(embed_dim // factor, embed_dim)
        self.norm = nn.BatchNorm1d(embed_dim)
    
    def get_knn(self, x, k=16):
        # x shape: (B, C, N)
        B, C, N = x.shape
        x = x.permute(0, 2, 1)  # 转为 (B, N, C)
        dist = torch.cdist(x, x)  # (B, N, N) 计算两点之间的距离
        knn_idx = dist.topk(k, largest=False, dim=-1).indices  # 最近的K个点索引
        return knn_idx

    def forward(self, x):
        B, C, N = x.shape
        knn_idx = self.get_knn(x, self.k_neighbors)  # 获取每个点的近邻
        x = x.permute(2, 0, 1)  # 转换为 (N, B, C)
        
        # 计算Q, K, V
        Q = self.query_proj(x)  # (N, B, C // factor)
        K = self.key_proj(x)    # (N, B, C // factor)
        V = self.value_proj(x)  # (N, B, C // factor)
        
        # 计算局部注意力权重，仅在邻域内计算注意力
        attn_weights = torch.einsum("nbe,nbe->be", Q[knn_idx], K[knn_idx]) / (self.embed_dim // self.factor) ** 0.5
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # 应用注意力到V
        attn_output = torch.einsum("be,nbe->nbe", attn_weights, V[knn_idx])
        attn_output = self.out_proj(attn_output)  # 投影回原始嵌入维度
        
        # 转换回原始形状 (B, C, N)
        attn_output = attn_output.permute(1, 2, 0)
        
        # 残差连接和规范化
        x = self.norm(x.permute(1, 2, 0) + attn_output)
        
        return x

class BlockNeighborhoodAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, factor=4, block_size=1024, k_neighbors=16):
        super(BlockNeighborhoodAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.factor = factor
        self.block_size = block_size
        self.k_neighbors = k_neighbors

        # 投影层
        self.query_proj = nn.Linear(embed_dim, embed_dim // factor)
        self.key_proj = nn.Linear(embed_dim, embed_dim // factor)
        self.value_proj = nn.Linear(embed_dim, embed_dim // factor)
        self.out_proj = nn.Linear(embed_dim // factor, embed_dim)
        self.norm = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        B, C, N = x.shape
        num_blocks = N // self.block_size
        
        # 初始化输出张量
        output = torch.zeros_like(x)
        
        # 将点云划分为块
        for i in range(num_blocks):
            start = i * self.block_size
            end = start + self.block_size
            x_block = x[:, :, start:end]  # (B, C, block_size)
            
            # 降维后的 Q, K, V
            Q = self.query_proj(x_block.permute(2, 0, 1))  # (block_size, B, C // factor)
            K = self.key_proj(x_block.permute(2, 0, 1))    # (block_size, B, C // factor)
            V = self.value_proj(x_block.permute(2, 0, 1))  # (block_size, B, C // factor)

            # 使用块内的 KNN
            knn_idx = self.get_knn_within_block(Q, self.k_neighbors)

            # 仅在局部邻域计算注意力权重
            attn_weights = torch.einsum("nbe,nbe->be", Q[knn_idx], K[knn_idx]) / (self.embed_dim // self.factor) ** 0.5
            attn_weights = F.softmax(attn_weights, dim=-1)
            
            # 将注意力权重应用到 V
            attn_output = torch.einsum("be,nbe->nbe", attn_weights, V[knn_idx])
            attn_output = self.out_proj(attn_output)
            
            # 转回原始形状 (B, C, block_size)
            attn_output = attn_output.permute(1, 2, 0)
            output[:, :, start:end] = attn_output

        # 残差连接和规范化
        output = self.norm(x + output)
        
        return output
    
    def get_knn_within_block(self, Q, k):
        """
        在降维后的局部块内进行 KNN 计算，避免在全局空间构建邻域矩阵。
        """
        block_size, B, C = Q.shape
        # 使用局部计算，生成 KNN 的索引
        # 假设使用近似 KNN 方法或 GPU 加速 KNN 查找来降低显存需求
        dist = torch.cdist(Q.permute(1, 0, 2), Q.permute(1, 0, 2))
        knn_idx = dist.topk(k, largest=False, dim=-1).indices
        return knn_idx

class RandomSamplingAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, factor=4, sample_size=512):
        super(RandomSamplingAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.factor = factor
        self.sample_size = sample_size  # 采样邻居数量

        # 投影层
        self.query_proj = nn.Linear(embed_dim, embed_dim // factor)
        self.key_proj = nn.Linear(embed_dim, embed_dim // factor)
        self.value_proj = nn.Linear(embed_dim, embed_dim // factor)
        self.out_proj = nn.Linear(embed_dim // factor, embed_dim)
        self.norm = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        B, C, N = x.shape
        x = x.permute(2, 0, 1)

        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)
        
        # 使用随机采样来选择部分邻居进行注意力计算
        sampled_idx = torch.randint(0, N, (N, self.sample_size)).to(x.device)  # (N, sample_size)

        attn_weights = torch.einsum("nbe,nbe->be", Q[sampled_idx], K[sampled_idx]) / (self.embed_dim // self.factor) ** 0.5
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        attn_output = torch.einsum("be,nbe->nbe", attn_weights, V[sampled_idx])
        attn_output = self.out_proj(attn_output)
        attn_output = attn_output.permute(1, 2, 0)

        output = self.norm(x.permute(1, 2, 0) + attn_output)

        return output

def approx_knn(x, k):
    # 快速近似邻域查找，例如使用 LSH 或 Faiss
    # 这里为简化起见，用随机邻居替代实际实现
    B, C, N = x.shape
    return torch.randint(0, N, (N, k)).to(x.device)  # 随机生成近似邻居索引

class ApproximateNeighborAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, factor=4, k_neighbors=16):
        super(ApproximateNeighborAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.factor = factor
        self.k_neighbors = k_neighbors

        self.query_proj = nn.Linear(embed_dim, embed_dim // factor)
        self.key_proj = nn.Linear(embed_dim, embed_dim // factor)
        self.value_proj = nn.Linear(embed_dim, embed_dim // factor)
        self.out_proj = nn.Linear(embed_dim // factor, embed_dim)
        self.norm = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        B, C, N = x.shape
        x = x.permute(2, 0, 1)

        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)
        
        # 使用近似邻居
        knn_idx = approx_knn(Q.permute(1, 2, 0), self.k_neighbors)
        
        attn_weights = torch.einsum("nbe,nbe->be", Q[knn_idx], K[knn_idx]) / (self.embed_dim // self.factor) ** 0.5
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        attn_output = torch.einsum("be,nbe->nbe", attn_weights, V[knn_idx])
        attn_output = self.out_proj(attn_output)
        attn_output = attn_output.permute(1, 2, 0)

        output = self.norm(x.permute(1, 2, 0) + attn_output)
        
        return output

class FactorizedSelfAttentionOptimized(nn.Module):
    def __init__(self, embed_dim, num_heads, factor=4):
        super(FactorizedSelfAttentionOptimized, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.factor = factor
        
        # 确保 embed_dim 可以被 factor 整除
        assert embed_dim % factor == 0, "Embedding dimension must be divisible by the factor."
        
        # 投影层，将Q, K, V映射到降维空间
        self.query_proj = nn.Linear(embed_dim, embed_dim // factor)
        self.key_proj = nn.Linear(embed_dim, embed_dim // factor)
        self.value_proj = nn.Linear(embed_dim, embed_dim // factor)
        
        # 输出投影回原始维度
        self.out_proj = nn.Linear(embed_dim // factor, embed_dim)
        
        # 规范化层
        self.norm = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        # x shape: (B, C, N)
        B, C, N = x.shape
        
        # 转换为 (N, B, C) 以便进行计算
        x = x.permute(2, 0, 1)  # 转换为 (N, B, C)
        
        # 计算Q, K, V
        Q = self.query_proj(x)  # (N, B, C // factor)
        K = self.key_proj(x)    # (N, B, C // factor)
        V = self.value_proj(x)  # (N, B, C // factor)
        
        # 计算注意力权重
        attn_weights = torch.einsum("nbe,nbe->be", Q, K) / (self.embed_dim // self.factor) ** 0.5
        attn_weights = F.softmax(attn_weights, dim=-1)  # 使用softmax进行归一化
        
        # 将注意力权重应用到V
        attn_output = torch.einsum("be,nbe->nbe", attn_weights, V)
        
        # 将attn_output投影回原始嵌入维度
        attn_output = self.out_proj(attn_output)  # (N, B, C)
        
        # 转换回原始形状 (B, C, N)
        attn_output = attn_output.permute(1, 2, 0)
        
        # 残差连接和规范化
        x = x.permute(1, 2, 0)  # 还原 x 的形状 (B, C, N)
        out = self.norm(x + attn_output)
        
        return out