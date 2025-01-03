import torch
import torch.nn.functional as F
from math import pi
import cv2 as cv
import kornia as K

from superpoint.utils.tools import dict_update


homography_adaptation_default_config = {
        'num': 1,
        'aggregation': 'sum',
        'valid_border_margin': 3,
        'homographies': {
            'translation': True,
            'rotation': True,
            'scaling': True,
            'perspective': True,
            'scaling_amplitude': 0.1,
            'perspective_amplitude_x': 0.1,
            'perspective_amplitude_y': 0.1,
            'patch_ratio': 0.5,
            'max_angle': pi,
        },
        'filter_counts': 0
}


def homography_adaptation(image, net, config):
    """执行单应性自适应。
    使用同一输入图像的多个随机变形块进行推理，以获得稳健的预测。
    
    参数:
        image: 形状为[N, H, W, 1]的张量。
        net: 一个函数，接受图像作为输入，执行推理，并输出预测字典。
        config: 配置字典，包含可选项如采样单应性数量'num'，聚合方法'aggregation'等。
    
    返回:
        包含预测概率的字典。
    """
    device = image.device
    probs = net(image)['prob']
    counts = torch.ones_like(probs)
    images = image

    probs = probs.unsqueeze(-1)
    counts = counts.unsqueeze(-1)
    images = images.unsqueeze(-1)

    shape = image.shape[1:3]
    config = dict_update(homography_adaptation_default_config, config)

    for i in range(config['num'] - 1):
        # 采样图像块
        H = sample_homography(shape, **config['homographies'])
        H = H.to(device)
        H_inv = invert_homography(H)
        
        # 使用kornia进行图像变换
        warped = K.geometry.warp_perspective(image.permute(0,3,1,2), 
                                           flat2mat(H), 
                                           (shape[0], shape[1]), 
                                           mode='bilinear').permute(0,2,3,1)
        
        count = K.geometry.warp_perspective(torch.ones_like(image).permute(0,3,1,2), 
                                          flat2mat(H_inv), 
                                          (shape[0], shape[1]), 
                                          mode='nearest').permute(0,2,3,1)
        
        mask = K.geometry.warp_perspective(torch.ones_like(image).permute(0,3,1,2), 
                                         flat2mat(H), 
                                         (shape[0], shape[1]), 
                                         mode='nearest').permute(0,2,3,1)

        # 忽略太靠近边界的检测以避免伪影
        if config['valid_border_margin']:
            kernel = cv.getStructuringElement(
                cv.MORPH_ELLIPSE, (config['valid_border_margin'] * 2,) * 2)
            kernel = torch.from_numpy(kernel).float().to(device)
            
            # 使用kornia进行腐蚀操作，确保维度正确
            count = count.permute(0,3,1,2)  # [B,H,W,C] -> [B,C,H,W]
            count_eroded = K.morphology.erosion(count, kernel) + 1.0  # 保持4维格式
            count = count_eroded.permute(0,2,3,1)  # [B,C,H,W] -> [B,H,W,C]
            
            mask = mask.permute(0,3,1,2)  # [B,H,W,C] -> [B,C,H,W]
            mask_eroded = K.morphology.erosion(mask, kernel) + 1.0  # 保持4维格式
            mask = mask_eroded.permute(0,2,3,1)  # [B,C,H,W] -> [B,H,W,C]

        # 预测检测概率
        prob = net(warped)['prob']
        prob = prob * mask
        
        prob_proj = K.geometry.warp_perspective(prob.unsqueeze(-1).permute(0,3,1,2), 
                                              flat2mat(H_inv), 
                                              (shape[0], shape[1]), 
                                              mode='bilinear').permute(0,2,3,1).squeeze(-1)
        prob_proj = prob_proj * count.squeeze(-1)

        probs = torch.cat([probs, prob_proj.unsqueeze(-1)], dim=-1)
        counts = torch.cat([counts, count], dim=-1)
        images = torch.cat([images, warped.unsqueeze(-1)], dim=-1)

    counts = torch.sum(counts, dim=-1)
    max_prob = torch.max(probs, dim=-1)[0]
    mean_prob = torch.sum(probs, dim=-1) / counts

    if config['aggregation'] == 'max':
        prob = max_prob
    elif config['aggregation'] == 'sum':
        prob = mean_prob
    else:
        raise ValueError(f'Unknown aggregation method: {config["aggregation"]}')

    if config['filter_counts']:
        prob = torch.where(counts >= config['filter_counts'],
                          prob, torch.zeros_like(prob))

    return {'prob': prob, 'counts': counts,
            'mean_prob': mean_prob, 'input_images': images, 'H_probs': probs}


def sample_homography(
        shape, perspective=True, scaling=True, rotation=True, translation=True,
        n_scales=5, n_angles=25, scaling_amplitude=0.1, perspective_amplitude_x=0.1,
        perspective_amplitude_y=0.1, patch_ratio=0.5, max_angle=pi/2,
        allow_artifacts=False, translation_overflow=0.):
    """采样随机有效单应性变换。

    计算原始图像中的随机块与相同图像大小的变形投影之间的单应性变换。
    与kornia.geometry.transform一样，它将输出（变形块）映射到变形后的输入点（原始块）。
    块（初始化为单的半尺寸中心裁剪）被迭代地投影、缩放、旋转和平移。

    参数:
        shape: 指定原始图像高度和宽度的二维张量。
        perspective: 布尔值，启用透视和仿射变换。
        scaling: 布尔值，启用块的随机缩放。
        rotation: 布尔值，启用块的随机旋转。
        translation: 布尔值，启用块的随机平移。
        n_scales: 缩放时采样的尝试尺度数量。
        n_angles: 旋转时采样的尝试角度数量。
        scaling_amplitude: 控制缩放量。
        perspective_amplitude_x: 控制x方向的透视效果。
        perspective_amplitude_y: 控制y方向的透视效果。
        patch_ratio: 控制用于创建单应性的块大小。
        max_angle: 旋转中使用的最大角度。
        allow_artifacts: 布尔值，启用应用单应性时的伪影。
        translation_overflow: 平移引起的边界伪影量。

    返回:
        形状为[8]的张量，对应于展平的单应性变换。
    """
    # 输出图像的角点
    margin = (1 - patch_ratio) / 2
    pts1 = margin + torch.tensor([[0, 0], [0, patch_ratio],
                                 [patch_ratio, patch_ratio], [patch_ratio, 0]],
                                dtype=torch.float32)
    # 输入块的角点
    pts2 = pts1.clone()

    # 随机透视和仿射扰动
    if perspective:
        if not allow_artifacts:
            perspective_amplitude_x = min(perspective_amplitude_x, margin)
            perspective_amplitude_y = min(perspective_amplitude_y, margin)
        perspective_displacement = torch.normal(0., perspective_amplitude_y/2, size=(1,))
        h_displacement_left = torch.normal(0., perspective_amplitude_x/2, size=(1,))
        h_displacement_right = torch.normal(0., perspective_amplitude_x/2, size=(1,))
        pts2 += torch.stack([
            torch.cat([h_displacement_left, perspective_displacement]),
            torch.cat([h_displacement_left, -perspective_displacement]),
            torch.cat([h_displacement_right, perspective_displacement]),
            torch.cat([h_displacement_right, -perspective_displacement])
        ])

    # 随机缩放
    if scaling:
        scales = torch.cat([torch.ones(1),
                          torch.normal(1, scaling_amplitude/2, size=(n_scales,))])
        center = torch.mean(pts2, dim=0, keepdim=True)
        scaled = (pts2 - center).unsqueeze(0) * scales.view(-1, 1, 1) + center
        if allow_artifacts:
            valid = torch.arange(1, n_scales + 1)  # 除scale=1外所有尺度都有效
        else:
            valid = torch.where(torch.all((scaled >= 0.) & (scaled <= 1.), dim=1).all(dim=1))[0]
            if len(valid) == 0:  # 如果没有有效的缩放，使用原始尺度
                valid = torch.tensor([0])
        idx = valid[torch.randint(len(valid), (1,))]
        pts2 = scaled[idx].squeeze(0)

    # 随机平移
    if translation:
        t_min = torch.min(pts2, dim=0)[0]
        t_max = torch.min(1 - pts2, dim=0)[0]
        if allow_artifacts:
            t_min += translation_overflow
            t_max += translation_overflow
        pts2 += torch.stack([
            torch.rand(1) * (t_max[0] - t_min[0]) + t_min[0],
            torch.rand(1) * (t_max[1] - t_min[1]) + t_min[1]
        ]).view(1, 2)

    # 随机旋转
    if rotation:
        angles = torch.linspace(-max_angle, max_angle, n_angles)
        angles = torch.cat([torch.zeros(1), angles])  # 以防没有有效旋转
        center = torch.mean(pts2, dim=0, keepdim=True)
        rot_mat = torch.stack([
            torch.cos(angles), -torch.sin(angles),
            torch.sin(angles), torch.cos(angles)
        ], dim=1).view(-1, 2, 2)
        rotated = torch.matmul(
            (pts2 - center).unsqueeze(0).expand(n_angles+1, -1, -1),
            rot_mat
        ) + center
        if allow_artifacts:
            valid = torch.arange(1, n_angles + 1)  # 除angle=0外所有角度都有效
        else:
            valid = torch.where(torch.all((rotated >= 0.) & (rotated <= 1.), dim=1).all(dim=1))[0]
            if len(valid) == 0:  # 如果没有有效的旋转，使用原始角度
                valid = torch.tensor([0])
        idx = valid[torch.randint(len(valid), (1,))]
        pts2 = rotated[idx].squeeze(0)

    # 缩放到实际大小
    shape = torch.tensor([shape[1], shape[0]], dtype=torch.float32)  # 不同的约定[y, x]
    pts1 = pts1 * shape.view(1, 2)
    pts2 = pts2 * shape.view(1, 2)

    def ax(p, q): return torch.tensor([p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]], dtype=torch.float32)
    def ay(p, q): return torch.tensor([0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]], dtype=torch.float32)

    a_mat = torch.stack([f(pts1[i], pts2[i]) for i in range(4) for f in (ax, ay)])
    p_mat = torch.tensor([[pts2[i][j] for i in range(4) for j in range(2)]], dtype=torch.float32).t()
    
    # 使用PyTorch的最小二乘求解器
    homography = torch.linalg.lstsq(a_mat, p_mat).solution
    # 确保返回8维向量
    return homography.squeeze(1)  # 返回[8]形状的张量


def invert_homography(H):
    """计算展平的单应性变换的逆变换。"""
    return mat2flat(torch.inverse(flat2mat(H)))


def flat2mat(H):
    """将形状为[B, 8]的展平单应性变换转换为形状为[B, 3, 3]的对应单应性矩阵。"""
    if H.dim() == 1:
        H = H.unsqueeze(0)  # 添加batch维度
    H = torch.cat([H, torch.ones(H.shape[0], 1, device=H.device)], dim=1)
    return H.view(-1, 3, 3)


def mat2flat(H):
    """将形状为[B, 3, 3]的单应性矩阵转换为其对应的展平形式[B, 8]。"""
    if H.dim() == 2:
        H = H.unsqueeze(0)  # 添加batch维度
    H = H.view(-1, 9)
    return H[:, :8] / H[:, 8:9]  # 使用广播除法


def compute_valid_mask(image_shape, homography, erosion_radius=0):
    """计算变换后的有效掩码。
    
    参数:
        image_shape: 原始图像的形状[H, W]。
        homography: 单应性变换矩阵。
        erosion_radius: 腐蚀半径，用于移除边界伪影。
    
    返回:
        有效区域的二值掩码。
    """
    mask = torch.ones(1, 1, image_shape[0], image_shape[1], device=homography.device)
    # 陈雪岩
    mask = K.geometry.warp_perspective(mask, 
                                     flat2mat(homography), 
                                     (image_shape[0], image_shape[1]), 
                                     mode='nearest')
    
    if erosion_radius > 0:
        # 创建2D kernel
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (erosion_radius * 2,) * 2)
        kernel = torch.from_numpy(kernel).float().to(mask.device)
        # kornia要求输入是4维的(B,C,H,W)，kernel是2维的
        mask = K.morphology.erosion(mask, kernel)  # 保持4维格式
    
    return mask.squeeze()  # 最后再squeeze


def warp_points(points, homography):
    """使用单应性变换对点进行变换。
    
    参数:
        points: 形状为[N, 2]的点坐标张量，坐标格式为(y,x)。
        homography: 单应性变换矩阵。
    
    返回:
        变换后的点坐标，保持(y,x)格式。
    """
    H = flat2mat(homography)[0]  # 获取3x3矩阵
    points = points[:, [1, 0]]
    # 转换为齐次坐标，注意保持(y,x)顺序
    points_h = torch.cat([points, torch.ones(points.shape[0], 1, device=points.device)], dim=1)
    
    # 应用变换
    warped_points = torch.matmul(H, points_h.t()).t()
    
    # 转换回欧几里得坐标
    warped_points = warped_points[:, :2] / warped_points[:, 2:3]
    warped_points = warped_points[:, [1, 0]]
    return warped_points  # 返回的点仍然保持(y,x)格式


def filter_points(points, shape):
    """过滤掉图像边界外的点。
    
    参数:
        points: 形状为[N, 2]的点坐标张量。
        shape: 图像形状[H, W]。
    
    返回:
        在图像边界内的点的掩码。
    """
    mask = (points[:, 0] >= 0) & (points[:, 0] <= shape[1]-1) & \
           (points[:, 1] >= 0) & (points[:, 1] <= shape[0]-1)
    return mask 