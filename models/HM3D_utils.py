import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from einops import rearrange
import sys,os
#from mamba_ssm.modules.mamba_simple import Mamba,GlobularMamba
#sys.path.append(os.path.abspath('/home/data_disk/cty/pyWorkSpace/3DM2'))
from mamba_ssm.modules.mamba_simple import GlobularMamba

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

# def reorder_fps(centroids, xyz):
#     """
#     对 FPS 采样后的点进行重新排序，使得相邻两点之间的距离最短
#
#     Input:
#         centroids: FPS采样的点的索引，形状为 [B, npoint]
#         xyz: 点云数据，形状为 [B, N, 3]
#
#     Output:
#         reordered_centroids: 排序后的采样点索引，形状为 [B, npoint]
#     """
#     B, npoint = centroids.shape
#     reordered_centroids = centroids.clone()  # 深拷贝采样的索引
#
#     # 对每个样本，执行点排序
#     for b in range(B):
#         selected_points = xyz[b, centroids[b], :]  # 获取该批次的采样点坐标
#         distances = torch.cdist(selected_points, selected_points)  # 计算采样点之间的距离，形状为 [npoint, npoint]
#
#         # 从第一个点开始，重新排序
#         reordered_points = [0]  # 从第一个点开始
#         remaining_points = list(range(1, npoint))  # 其余的点
#
#         while remaining_points:
#             last_point = reordered_points[-1]  # 获取当前已排序的最后一个点
#             last_point_idx = selected_points[last_point]  # 当前点的坐标
#             # 计算当前点与剩下的所有点的距离
#             dist_to_last = distances[last_point, remaining_points]
#             # 选择距离最近的点
#             next_point_idx = remaining_points[torch.argmin(dist_to_last)]
#             reordered_points.append(next_point_idx)
#             remaining_points.remove(next_point_idx)
#
#         reordered_centroids[b] = centroids[b, reordered_points]  # 按照新的顺序更新索引
#
#     return reordered_centroids


# def reorder_fps(centroids, xyz):
#     """
#     对 FPS 采样后的点进行重新排序，使得相邻两点之间的距离最短
#
#     Input:
#         centroids: FPS采样的点的索引，形状为 [B, npoint]
#         xyz: 点云数据，形状为 [B, N, 3]
#
#     Output:
#         reordered_centroids: 排序后的采样点索引，形状为 [B, npoint]
#     """
#     B, npoint = centroids.shape
#     reordered_centroids = centroids.clone()  # 深拷贝采样的索引
#     device = xyz.device
#
#     # 对每个样本，执行点排序
#     for b in range(B):
#         # 获取该批次的采样点坐标
#         selected_points = xyz[b, centroids[b], :]
#
#         # 计算采样点之间的距离，形状为 [npoint, npoint]
#         distances = torch.cdist(selected_points, selected_points)  # 计算点对点的距离
#
#         reordered_points = torch.zeros(npoint, dtype=torch.long).to(device)  # 初始化排序后的点索引
#         remaining_points = torch.arange(1, npoint, dtype=torch.long).to(device)  # 剩余未排序的点
#
#         reordered_points[0] = 0  # 从第一个点开始
#
#         for i in range(1, npoint):
#             last_point_idx = reordered_points[i - 1]  # 当前已排序的最后一个点
#             # 计算当前点与剩下的所有点的距离
#             dist_to_last = distances[last_point_idx, remaining_points]
#
#             # 选择距离最近的点
#             next_point_idx = remaining_points[torch.argmin(dist_to_last)]  # 获取最小距离点的索引
#             reordered_points[i] = next_point_idx
#
#             # 更新 remaining_points 张量，避免 Python 列表的删除操作
#             remaining_points = remaining_points[remaining_points != next_point_idx]
#
#         reordered_centroids[b] = centroids[b, reordered_points]  # 按照新的顺序更新索引
#
#     return reordered_centroids
# 计算相邻两点之间的距离
def compute_distances(xyz, centroids):
    """
    计算已采样点之间的距离。

    Input:
        xyz: 点云数据，形状为 [B, N, 3]
        centroids: 采样的点的索引，形状为 [B, npoint]

    Output:
        distances: 每对相邻采样点之间的欧氏距离，形状为 [B, npoint-1]
    """
    B, npoint = centroids.shape
    distances = []

    for b in range(B):
        selected_points = xyz[b, centroids[b], :]  # 获取当前批次的采样点坐标
        dist = torch.norm(selected_points[1:] - selected_points[:-1], dim=-1)  # 计算相邻点的距离
        distances.append(dist)

    return torch.stack(distances, dim=0)  # 返回每批次的距离


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
    #print("before sorted dist_mask:",dist_mask)
    # 根据距离排序，返回排序后的索引和距离
    sorted_dist, sorted_idx = torch.sort(dist, dim=-1)  # sorted_dist: [B, S, N]，sorted_idx: [B, S, N]
    #print("sorted_idx:", sorted_idx)

    # 在半径范围内，设置距离大于 radius 的点为无效，超出范围的点索引设置为 N
    dist_mask = dist_mask.gather(-1, sorted_idx)  # dist_mask shape: [B, S, N], 用排序后的索引来选择mask
    #print("after sorted dist_mask:", dist_mask)
    # 将超出范围的点距离设置为无穷大
    sorted_dist[~dist_mask] = float('inf')  # 设置超出范围的点距离为无穷大

    # 选择前 k 个邻居
    group_idx = sorted_idx[:, :, :k]  # 选择前 k 个邻居

    #print("before group_idx:", group_idx)

    # 对于超出半径范围的点，将其索引设置为 N，并用范围内的第一个点的索引替代
    group_first = group_idx[:, :, 0].view(group_idx.shape[0], group_idx.shape[1], 1).repeat(1, 1, k)  # 第一个点的索引

    # 对超出范围的点，替换为第一个点的索引
    # 扩展 dist_mask，使得它可以与 group_first 对齐
    dist_mask_expanded = dist_mask[:, :, :k]  # 我们只需要前 k 个点的 dist_mask 来替换
    group_idx[~dist_mask_expanded] = group_first[~dist_mask_expanded]  # 仅替换超出范围的点的索引

    #print("after group_idx:", group_idx)

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
    #fps_idx = reorder_fps(fps_idx, xyz)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point_with_sort(radius, nsample, xyz, new_xyz)
    #idx = query_ball_point(radius, nsample, xyz, new_xyz)
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


def sample_and_group_density(npoint, radius, nsample, xyz, points, density_scale = None):
    """
    Input:
        npoint:
        nsample:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point_with_sort(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    if density_scale is None:
        return new_xyz, new_points, grouped_xyz_norm, idx
    else:
        grouped_density = index_points(density_scale, idx)
        return new_xyz, new_points, grouped_xyz_norm, idx, grouped_density

def sample_and_group_all_density(xyz, points, density_scale = None):
    """
    Input:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    #new_xyz = torch.zeros(B, 1, C).to(device)
    new_xyz = xyz.mean(dim = 1, keepdim = True)
    grouped_xyz = xyz.view(B, 1, N, C) - new_xyz.view(B, 1, 1, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    if density_scale is None:
        return new_xyz, new_points, grouped_xyz
    else:
        grouped_density = density_scale.view(B, 1, N, 1)
        return new_xyz, new_points, grouped_xyz, grouped_density

def compute_density(xyz, bandwidth):
    '''
    xyz: input points position data, [B, N, C]
    '''
    #import ipdb; ipdb.set_trace()
    B, N, C = xyz.shape
    sqrdists = square_distance(xyz, xyz)
    gaussion_density = torch.exp(- sqrdists / (2.0 * bandwidth * bandwidth)) / (2.5 * bandwidth)
    xyz_density = gaussion_density.mean(dim = -1)

    return xyz_density


class DensityNet(nn.Module):
    def __init__(self, hidden_unit=[16, 8]):
        super(DensityNet, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        self.mlp_convs.append(nn.Conv2d(1, hidden_unit[0], 1))
        self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
        for i in range(1, len(hidden_unit)):
            self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
        self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], 1, 1))
        self.mlp_bns.append(nn.BatchNorm2d(1))

    def forward(self, density_scale):
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            density_scale = bn(conv(density_scale))
            if i == len(self.mlp_convs):
                density_scale = F.sigmoid(density_scale)
            else:
                density_scale = F.relu(density_scale)

        return density_scale


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

class GlobularMambaDensitySetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, group_all, is_emd, bandwidth=0.1):
        super(GlobularMambaDensitySetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.is_emd=is_emd
        self.bandwidth = bandwidth
        self.densitynet = DensityNet()
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
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1) # B L C
        if points is not None:
            points = points.permute(0, 2, 1)

        xyz_density = compute_density(xyz, self.bandwidth)
        inverse_density = 1.0 / xyz_density

        if self.group_all:
            new_xyz, new_points, grouped_xyz_norm, grouped_density = sample_and_group_all_density(xyz, points, inverse_density.view(B, N, 1))
        else:
            new_xyz, new_points, grouped_xyz_norm, _, grouped_density = sample_and_group_density(self.npoint, self.radius, self.nsample, xyz, points, inverse_density.view(B, N, 1))

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


        inverse_max_density = grouped_density.max(dim = 2, keepdim=True)[0]
        density_scale = grouped_density / inverse_max_density
        #print("density_scale:", density_scale.shape)
        #density_scale = self.densitynet(density_scale.permute(0, 3, 2, 1))
        #density_scale = density_scale.permute(0, 3, 2, 1)
        # 1. 扩展 density_scale 到 [b, n, k, c]，让每个邻居的权重与对应特征的每个维度匹配
        expanded_density_scale = density_scale.expand(-1, -1, -1, new_points.shape[-1])
        # print("expanded_density_scale:",expanded_density_scale.shape)

        # # 2. 对每个采样点的邻居特征加权求和
        # weighted_sum = (new_points * expanded_density_scale).sum(dim=2)
        # # 3. 对加权和进行归一化，使用权重的和进行归一化
        # # 对每个采样点的邻居的权重进行求和
        # sum_of_weights = expanded_density_scale.sum(dim=2)
        #
        # # 如果有权重和为零的情况，避免除零错误（可以根据需求设置为0或其他值）
        # sum_of_weights = sum_of_weights + 1e-6
        #
        # # 进行归一化
        # new_points = weighted_sum / sum_of_weights

        # 2. 对每个采样点的邻居特征进行最大池化
        # 使用 torch.max 来对特征维度进行最大池化
        new_points = torch.max(new_points * expanded_density_scale,2)[0]
        #new_points= (new_points * expanded_density_scale).max(dim=2)[0] # 对每个采样点的邻居特征做最大池化

        #print("new_points:", new_points.shape)

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
    model = GlobularMambaDensitySetAbstraction(npoint=npoint, radius=radius, nsample=nsample,
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

# 测试代码
def test_fps_reorder():
    # 生成一个随机的点云数据 [B, N, 3]
    B = 2  # 批量大小
    N = 10  # 点的数量
    xyz = torch.randn(B, N, 3).to(torch.device('cuda'))  # 使用 GPU 生成随机点云数据

    npoint = 5  # 需要采样的点的数量
    centroids = farthest_point_sample(xyz, npoint)  # 使用 FPS 采样点
    reordered_centroids = reorder_fps(centroids, xyz)  # 对采样点进行重新排序

    print("原始采样点索引（FPS采样）：")
    print(centroids)
    print("重新排序后的采样点索引（相邻两点距离最短）：")
    print(reordered_centroids)

    # 打印排序前后的相邻点距离
    print("\n--- 排序前的相邻点之间的距离 ---")
    distances_before = compute_distances(xyz, centroids)
    print(distances_before)

    print("\n--- 排序后的相邻点之间的距离 ---")
    distances_after = compute_distances(xyz, reordered_centroids)
    print(distances_after)

if __name__ == "__main__":
    main()
