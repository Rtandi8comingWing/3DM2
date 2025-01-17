import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from einops import rearrange
import sys,os
#from mamba_ssm.modules.mamba_simple import Mamba,GlobularMamba
#sys.path.append(os.path.abspath('/home/data_disk/cty/pyWorkSpace/3DM2'))
#from mamba_ssm.modules.mamba_simple import GlobularMamba

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


def query_ball_point_with_sort(radius, k, xyz, new_xyz):
    """
    查找每个查询点半径内的 k 个最近邻，并按距离从近到远排序

    Input:
        radius: 半径范围
        k: 要选择的邻居个数
        xyz: 所有点的坐标 [B, N, C]
        new_xyz: 查询点的坐标 [B, S, C]

    Output:
        group_idx: 排序后的邻居索引，形状为 [B, S, k]
    """
    # 计算每个查询点与所有点的平方距离
    dist = square_distance(new_xyz, xyz)  # dist shape: [B, S, N]

    # 选择在半径范围内的点
    dist_mask = dist <= radius ** 2  # dist_mask shape: [B, S, N]，表示哪些点在半径范围内

    # 根据距离排序，返回排序后的索引和距离
    sorted_dist, sorted_idx = torch.sort(dist, dim=-1)  # sorted_dist: [B, S, N]，sorted_idx: [B, S, N]

    # 在半径范围内，设置距离大于 radius 的点为无效，超出范围的点索引设置为 N
    dist_mask = dist_mask.gather(-1, sorted_idx)  # dist_mask shape: [B, S, N], 用排序后的索引来选择mask

    # 将超出范围的点距离设置为无穷大
    sorted_dist[~dist_mask] = float('inf')  # 设置超出范围的点距离为无穷大

    # 选择前 k 个邻居
    group_idx = sorted_idx[:, :, :k]  # 选择前 k 个邻居

    # 对于超出半径范围的点，将其索引设置为 N，并用范围内的第一个点的索引替代
    group_first = group_idx[:, :, 0].view(group_idx.shape[0], group_idx.shape[1], 1).repeat(1, 1, k)  # 第一个点的索引

    # 找到超出范围的点，替换为 N (N是索引的最大值，超出索引范围)
    mask_out_of_range = (dist_mask.sum(dim=-1) < k)  # 如果在半径范围内的点少于k个，表示有点超出范围
    group_idx[mask_out_of_range] = group_first[mask_out_of_range]  # 替换超出范围的点索引

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


class GlobularMambaSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, group_all, is_emd):
        super(GlobularMambaSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.is_emd=is_emd
        if self.is_emd:
            self.mamba_inchannel = 64
            self.emd = nn.Linear(in_channel, self.mamba_inchannel)
        else:
            self.mamba_inchannel = in_channel
            self.emd = None
        self.mamba = GlobularMamba(self.mamba_inchannel)
    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1) # B L C
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        B,N,K,C = new_points.shape
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        # new_points = rearrange(new_points, 'b n k c -> (b n) k c')  # 合并 K 和 C 维度
        # new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        if self.is_emd:
            new_points = rearrange(new_points, 'b n k c -> (b n k) c')  # 合并 K 和 C 维度
            new_points = self.emd(new_points)
            new_points = rearrange(new_points, '(b n k) c -> (b n) k c', b=B, n=N, k=K)  # 合并 K 和 C 维度
        else :
            new_points = rearrange(new_points, 'b n k c -> (b n) k c')  # 合并 K 和 C 维度

        new_points = self.mamba(new_points)
        new_points = rearrange(new_points, '(b n) k c -> b n k c', b=B, n=N)  # 合并 K 和 C 维度
        new_points = torch.max(new_points, 2)[0] #B, N, C
        new_xyz = new_xyz.permute(0, 2, 1) #B, 3, N
        new_points = new_points.permute(0, 2, 1) #B, C, N
        return new_xyz, new_points



# 主函数
def forward():
    # 模拟输入参数
    B = 8       # 批次大小
    C = 3       # xyz 坐标的通道数
    D = 6       # 每个点的特征维度
    N = 1024    # 点的数量
    npoint = 512  # 采样点的数量
    radius = 0.2  # 半径（不用于示例中）
    nsample = 32  # 邻居的数量
    group_all = False  # 是否对所有点进行分组
    is_emd = True   # 是否使用 EMD 层

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # 创建模型
    model = GlobularMambaSetAbstraction(npoint=npoint, radius=radius, nsample=nsample,
                                        in_channel=C, group_all=group_all, is_emd=is_emd)

    model = model.to(device)  # 确保模型在 GPU 上
    # 创建模拟数据
    xyz = torch.randn(B, C, N)  # 点云位置，形状 [B, C, L] 8 3 1024
    # points = torch.randn(B, D, N)  # 点特征，形状 [B, D, N]

    points = None
    xyz = xyz.to(device)
    # 前向传播
    new_xyz, new_points = model(xyz, points)

    # 输出结果
    print("Output new_xyz shape:", new_xyz.shape)  # 应该是 [B, npoint, C] 8 512 3
    print("Output new_points shape:", new_points.shape)  # 应该是 [B, npoint, mamba_inchannel] 8*512 32 128   (b n) k c


# 主函数
def main():
    # 模拟输入参数
    B = 8  # 批次大小
    C = 3  # xyz 坐标的通道数
    D = 6  # 每个点的特征维度
    N = 1024  # 点的数量
    npoint = 512  # 采样点的数量
    radius = 0.2  # 半径（不用于示例中）
    nsample = 32  # 邻居的数量
    group_all = False  # 是否对所有点进行分组
    is_emd = True  # 是否使用 EMD 层

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型
    model = GlobularMambaSetAbstraction(npoint=npoint, radius=radius, nsample=nsample,
                                        in_channel=C, group_all=group_all, is_emd=is_emd)

    model = model.to(device)  # 确保模型在 GPU 上

    # 创建模拟数据
    xyz = torch.randn(B, C, N, requires_grad=True)  # 点云位置，形状 [B, C, N]
    points = torch.randn(B, D, N, requires_grad=True)  # 点特征，形状 [B, D, N]


    points = None
    xyz = xyz.to(device)
    # 前向传播
    new_xyz, new_points = model(xyz, points)

    # 输出结果形状检查
    print("Output new_xyz shape:", new_xyz.shape)  # 应该是 [B, npoint, C]
    print("Output new_points shape:", new_points.shape)  # 应该是 [B, npoint, mamba_inchannel]

    # 构造一个简单的损失函数，用于后向传播测试
    loss_fn = nn.MSELoss()

    # 假设我们的目标是 new_points 的最后一维（例如，进行回归任务）
    target = torch.randn_like(new_points)  # 目标值，形状与 new_points 相同

    # 计算损失
    loss = loss_fn(new_points, target)
    print("Loss:", loss.item())

    # 后向传播
    loss.backward()

    # 检查梯度
    #print("Gradient of xyz:", xyz.grad)  # xyz 的梯度
    #print("Gradient of points:", points.grad)  # points 的梯度


# 验证 query_ball_point_with_sort 函数的效果
def validate_query_ball_point_with_sort():
    # 设置随机种子，确保结果可复现
    torch.manual_seed(3)

    # 模拟数据：生成随机点坐标
    B, N, C = 2, 10, 3  # 批次大小、场景中点的数量、每个点的坐标维度
    S, k = 2, 3  # 查询点数量和选择的邻居数量
    radius = 1.5  # 半径范围

    # 生成场景点坐标和查询点坐标
    xyz = torch.randn(B, N, C)  # 场景中所有点的坐标
    new_xyz = torch.randn(B, S, C)  # 查询点的坐标

    print("场景点 xyz:\n", xyz)
    print("查询点 new_xyz:\n", new_xyz)

    # 调用 query_ball_point_with_sort 查找每个查询点的最近 k 个邻居
    group_idx = query_ball_point_with_sort(radius, k, xyz, new_xyz)

    print("\n查询点的最近邻索引（按距离从近到远排序）:\n", group_idx)

if __name__ == "__main__":
    validate_query_ball_point_with_sort()
