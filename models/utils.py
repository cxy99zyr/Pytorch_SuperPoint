import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia as K

from .homographies import warp_points
from .backbones.vgg import vgg_block


def detector_head(inputs, **config):
    """检测头实现。
    
    参数:
        inputs: 输入特征图，形状为[B, C, H, W]。
        config: 配置字典。
        
    返回:
        包含logits和概率的字典。
    """
    params_conv = {
        'padding': 'same',
        'batch_norm': config.get('batch_normalization', True),
        'training': config.get('training', True),
        'kernel_reg': config.get('kernel_reg', 0.)
    }
    
    x = vgg_block(inputs, 256, 3, 'conv1', 
                  activation=F.relu, **params_conv)
    x = vgg_block(x, 1+pow(config['grid_size'], 2), 1, 'conv2',
                  activation=None, **params_conv)
    
    # 计算概率
    prob = F.softmax(x, dim=1)
    # 移除"非关键点"的dustbin通道
    prob = prob[:, :-1]
    
    # 重新排列为网格点
    B, C, H, W = prob.shape
    prob = prob.view(B, config['grid_size'], 
                    config['grid_size'], H, W)
    prob = prob.permute(0, 3, 1, 4, 2).contiguous()
    prob = prob.view(B, H*config['grid_size'], 
                    W*config['grid_size'])
    
    return {'logits': x, 'prob': prob}


def descriptor_head(inputs, **config):
    """描述子头实现。
    
    参数:
        inputs: 输入特征图，形状为[B, C, H, W]。
        config: 配置字典。
        
    返回:
        包含原始描述子和归一化描述子的字典。
    """
    params_conv = {
        'padding': 'same',
        'batch_norm': config.get('batch_normalization', True),
        'training': config.get('training', True),
        'kernel_reg': config.get('kernel_reg', 0.)
    }
    
    x = vgg_block(inputs, 256, 3, 'conv1',
                  activation=F.relu, **params_conv)
    x = vgg_block(x, config['descriptor_size'], 1, 'conv2',
                  activation=None, **params_conv)
    
    # 上采样和归一化
    B, C, H, W = x.shape
    target_size = (H * config['grid_size'], W * config['grid_size'])
    desc = F.interpolate(x, size=target_size, mode='bilinear', 
                        align_corners=False)
    desc = F.normalize(desc, p=2, dim=1)
    
    return {'descriptors_raw': x, 'descriptors': desc}


def detector_loss(keypoint_map, logits, valid_mask=None, **config):
    """检测头损失函数。
    
    参数:
        keypoint_map: 关键点标签图。
        logits: 模型输出的logits。
        valid_mask: 有效区域掩码。
        config: 配置字典。
        
    返回:
        标量损失值。
    """
    # 准备标签
    labels = keypoint_map.unsqueeze(-1).float()
    labels = F.pixel_unshuffle(labels, config['grid_size'])
    shape = list(labels.shape)
    shape[-1] = 1
    labels = torch.cat([2*labels, torch.ones(*shape, device=labels.device)], -1)
    
    # 添加小的随机值以打破相等值的平局
    rand_break = torch.rand_like(labels) * 0.1
    labels = torch.argmax(labels + rand_break, dim=-1)
    
    # 处理有效掩码
    valid_mask = torch.ones_like(keypoint_map) if valid_mask is None else valid_mask
    valid_mask = valid_mask.unsqueeze(-1).float()
    valid_mask = F.pixel_unshuffle(valid_mask, config['grid_size'])
    valid_mask = valid_mask.prod(dim=-1)  # 在通道维度上进行AND操作
    
    # 计算交叉熵损失
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), 
                          labels.reshape(-1),
                          reduction='none')
    loss = (loss * valid_mask.reshape(-1)).mean()
    
    return loss


def descriptor_loss(descriptors, warped_descriptors, homographies,
                   valid_mask=None, **config):
    """描��子头损失函数。
    
    参数:
        descriptors: 原始图像的描述子。
        warped_descriptors: 变换后图像的描述子。
        homographies: 单应性变换矩阵。
        valid_mask: 有效区域掩码。
        config: 配置字典。
        
    返回:
        标量损失值。
    """
    # 计算每个单元格中心像素的位置
    B, C, Hc, Wc = descriptors.shape
    y, x = torch.meshgrid(torch.arange(Hc, device=descriptors.device),
                         torch.arange(Wc, device=descriptors.device))
    grid_size = config['grid_size']
    coord_cells = torch.stack([y, x], dim=-1) * grid_size + grid_size // 2
    
    # 计算变换后的中心像素位置
    coord_cells_flat = coord_cells.reshape(-1, 2).float()
    warped_coord_cells = warp_points(coord_cells_flat, homographies)
    warped_coord_cells = warped_coord_cells.reshape(B, Hc, Wc, 2)
    
    # 计算配对距离并过滤小于阈值的距离
    coord_cells = coord_cells.unsqueeze(0).unsqueeze(3).unsqueeze(4)
    warped_coord_cells = warped_coord_cells.unsqueeze(2).unsqueeze(2)
    cell_distances = torch.norm(coord_cells - warped_coord_cells, dim=-1)
    s = (cell_distances <= grid_size - 0.5).float()
    
    # 归一化描述子并计算配对点积
    descriptors = descriptors.reshape(B, C, Hc, Wc, 1, 1)
    descriptors = F.normalize(descriptors, p=2, dim=1)
    warped_descriptors = warped_descriptors.reshape(B, C, 1, 1, Hc, Wc)
    warped_descriptors = F.normalize(warped_descriptors, p=2, dim=1)
    dot_product_desc = torch.sum(descriptors * warped_descriptors, dim=1)
    dot_product_desc = F.relu(dot_product_desc)
    
    # 归一化点积
    dot_product_desc = dot_product_desc.reshape(B, Hc, Wc, -1)
    dot_product_desc = F.normalize(dot_product_desc, p=2, dim=-1)
    dot_product_desc = dot_product_desc.reshape(B, Hc, Wc, Hc, Wc)
    dot_product_desc = dot_product_desc.permute(0, 2, 1, 3, 4)
    dot_product_desc = dot_product_desc.reshape(B, Hc * Wc, -1)
    dot_product_desc = F.normalize(dot_product_desc, p=2, dim=1)
    dot_product_desc = dot_product_desc.reshape(B, Hc, Wc, Hc, Wc)
    
    # 计算损失
    positive_dist = torch.clamp(config['positive_margin'] - dot_product_desc, min=0.)
    negative_dist = torch.clamp(dot_product_desc - config['negative_margin'], min=0.)
    loss = config['lambda_d'] * s * positive_dist + (1 - s) * negative_dist
    
    # 处理有效掩码
    if valid_mask is None:
        valid_mask = torch.ones(B, Hc * grid_size, Wc * grid_size,
                              device=descriptors.device)
    valid_mask = valid_mask.unsqueeze(-1).float()
    valid_mask = F.pixel_unshuffle(valid_mask, grid_size)
    valid_mask = valid_mask.prod(dim=-1)
    valid_mask = valid_mask.reshape(B, 1, 1, Hc, Wc)
    
    normalization = valid_mask.sum() * (Hc * Wc)
    loss = (valid_mask * loss).sum() / normalization
    
    return loss


def spatial_nms(prob, size):
    """使用最大池化执行非最大抑制。
    
    参数:
        prob: 概率热图。
        size: 池化窗口大小。
        
    返回:
        经过NMS处理的概率图。
    """
    prob = prob.unsqueeze(0).unsqueeze(0)
    pooled = F.max_pool2d(prob, kernel_size=size, stride=1, padding=size//2)
    prob = torch.where(prob == pooled, prob, torch.zeros_like(prob))
    return prob.squeeze()


def box_nms(prob, size, iou=0.1, min_prob=0.01, keep_top_k=0):
    """使用边界框执行非最大抑制。
    
    参数:
        prob: 概率热图。
        size: 边界框大小。
        iou: IoU阈值。
        min_prob: 最小概率阈值。
        keep_top_k: 保留的最高分数数量。
        
    返回:
        经过NMS处理的概率图。
    """
    # 转换为边界框格式
    pts = torch.nonzero(prob >= min_prob, as_tuple=False).float()
    if pts.shape[0] == 0:
        return torch.zeros_like(prob)
        
    size = size / 2.
    boxes = torch.cat([pts - size, pts + size], dim=1)
    scores = prob[pts[:, 0].long(), pts[:, 1].long()]
    
    # 执行NMS
    keep = K.ops.nms(boxes, scores, iou)
    pts = pts[keep]
    scores = scores[keep]
    
    # 保留top-k
    if keep_top_k > 0:
        k = min(keep_top_k, scores.shape[0])
        scores, indices = torch.topk(scores, k)
        pts = pts[indices]
    
    # 重建概率图
    prob = torch.zeros_like(prob)
    prob[pts[:, 0].long(), pts[:, 1].long()] = scores
    
    return prob 