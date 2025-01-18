import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction
from models.HM3D_utils import GlobularMambaSetAbstraction, GlobularMambaDensitySetAbstraction
from mamba_ssm.modules.mamba_simple import Mamba,GlobularMamba


class HM3D(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(HM3D, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        # self.sa1 = GlobularMambaSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, group_all=False, is_emd=True)
        # self.sa2 = GlobularMambaSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, group_all=False, is_emd=False)
        # self.sa3 = GlobularMambaSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=262 + 3, group_all=True,is_emd=False)

        self.sa1 = GlobularMambaDensitySetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, group_all=False, is_emd=True, bandwidth= 0.1)
        self.sa2 = GlobularMambaDensitySetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, group_all=False, is_emd=False, bandwidth= 0.2)
        self.sa3 = GlobularMambaDensitySetAbstraction(npoint=None, radius=None, nsample=None, in_channel=262 + 3, group_all=True,is_emd=False, bandwidth= 0.4)

        self.fc1 = nn.Linear(530, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(128, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        # print("xyz shape:",xyz.shape)
        l1_xyz, l1_points = self.sa1(xyz, norm)
        # print("l1_points shape:B, C, N", l1_points.shape) #24, 128, 512
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # print("l2_points shape:B, C, N", l2_points.shape) #24, 262, 128
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # print("l3_points shape:B, C, N", l3_points.shape)  # 24, 530, 1
        x = l3_points.view(B, 530)
        # print("l3_points shape:B, C, N", l3_points.shape)  # 24, 530,
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)

        return x, l3_points



class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss