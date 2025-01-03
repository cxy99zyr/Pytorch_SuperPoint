import collections
import torch
import numpy as np


def dict_update(d, u):
    """Improved update for nested dictionaries.

    Arguments:
        d: The dictionary to be updated.
        u: The update dictionary.

    Returns:
        The updated dictionary.
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d 

def ensure_image_on_device(image, device):
    """确保图像在正确的设备上并具有正确的格式。

    参数:
        image: 输入图像，可以是 numpy 数组或 PyTorch 张量。
        device: 目标设备。

    返回:
        正确格式的 PyTorch 张量。
    """
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    
    if image.dim() == 2:  # 单通道图像
        image = image.unsqueeze(0)  # 添加通道维度
    elif image.dim() == 3 and image.shape[0] != 1:  # 通道在最后
        image = image.permute(2, 0, 1)  # [H,W,C] -> [C,H,W]
    
    if image.dim() == 3:
        image = image.unsqueeze(0)  # 添加批次维度
        
    return image.to(device)

def normalize_image(image):
    """标准化图像。

    参数:
        image: 输入图像张量，形状为 [B,C,H,W]。

    返回:
        标准化后的图像张量。
    """
    mean = image.mean(dim=[2, 3], keepdim=True)
    std = image.std(dim=[2, 3], keepdim=True)
    return (image - mean) / (std + 1e-7) 