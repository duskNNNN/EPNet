import torch.nn as nn
import torch
import torch.nn.functional as F
from models.pointnet2_utils_plus import PointNetSetAbstractionMsgSelectNormal,PointNetFeaturePropagation,FactorizedSelfAttention,SelfAttentionWithResidual


class get_model(nn.Module):
    def __init__(self, num_classes, normal_channel=False):
        super(get_model, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsgSelectNormal(512, [128, 256, 512], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsgSelectNormal(128, [128, 256], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstractionMsgSelectNormal(32, [128], 256+256, [[256, 512, 1024]])
        self.fp3 = PointNetFeaturePropagation(in_channel=1024+256+256, mlp=[1024, 512])
        self.fp2 = PointNetFeaturePropagation(in_channel=512+64+128+128, mlp=[512, 320])
        self.fp1 = PointNetFeaturePropagation(in_channel=320+6+1+additional_channel, mlp=[256, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)
        # 正则化
        self.bn_l2 = nn.BatchNorm1d(512) 
        self.bn_l1 = nn.BatchNorm1d(320) 
        self.bn_l0 = nn.BatchNorm1d(128)
        self.attention_l0 = FactorizedSelfAttention(embed_dim=128, num_heads=8)
        self.attention_l1 = SelfAttentionWithResidual(embed_dim=320, num_heads=8)
        self.attention_l2 = SelfAttentionWithResidual(embed_dim=512, num_heads=8)
        self.conv_l0 = nn.Conv1d(3+additional_channel, 128, 1)

    def forward(self, xyz, cls_label):
        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points_res = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        # l2_points = self.conv_l2(l2_points)
        l2_points = self.bn_l2(l2_points_res + l2_points)
        l2_points = self.attention_l2(l2_points)
        l1_points_res = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        # l1_points = self.conv_l1(l1_points)
        l1_points = self.bn_l1(l1_points_res + l1_points)
        l1_points = self.attention_l1(l1_points)
        # cls_label_one_hot = cls_label.view(B,16,1).repeat(1,1,N)
        cls_label_one_hot = cls_label.view(B,1,1).repeat(1,1,N)
        l0_points_res = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot,l0_xyz,l0_points],1), l1_points)
        # 更改通道数
        l0_points = self.conv_l0(l0_points)
        # 跳过连接
        l0_points = self.bn_l0(l0_points + l0_points_res)
        l0_points = self.attention_l0(l0_points)
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss