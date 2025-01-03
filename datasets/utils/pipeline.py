import torch
import cv2 as cv
import numpy as np
import kornia.geometry as kgm
import torchvision.transforms.functional as F

from superpoint.datasets.utils import photometric_augmentation as photaug
from superpoint.models.homographies import (sample_homography, compute_valid_mask,
                                            warp_points, filter_points, flat2mat,
                                            mat2flat)


def parse_primitives(names, all_primitives):
    """Parse the list of augmentation primitives.
    
    Args:
        names: String 'all' or list of primitive names.
        all_primitives: List of all available primitives.
        
    Returns:
        List of selected primitives.
    """
    p = all_primitives if (names == 'all') \
            else (names if isinstance(names, list) else [names])
    assert set(p) <= set(all_primitives)
    return p


def photometric_augmentation(data, **config):
    """Apply photometric augmentation to the image.
    
    Args:
        data: Dictionary containing the image and other data.
        config: Configuration dictionary.
        
    Returns:
        Updated data dictionary with augmented image.
    """
    primitives = parse_primitives(config['primitives'], photaug.augmentations)
    prim_configs = [config['params'].get(p, {}) for p in primitives]

    if config['random_order']:
        indices = torch.randperm(len(primitives))
    else:
        indices = torch.arange(len(primitives))

    image = data['image']
    for i in indices:
        aug_fn = getattr(photaug, primitives[i])
        image = aug_fn(image, **prim_configs[i])

    return {**data, 'image': image}


def homographic_augmentation(data, add_homography=False, **config):
    """Apply homographic augmentation to the data.
    
    Args:
        data: Dictionary containing the image and keypoints.
        add_homography: Whether to add the homography to the output.
        config: Configuration dictionary.
        
    Returns:
        Updated data dictionary with warped image and keypoints.
    """
    image_shape = torch.tensor(data['image'].shape[1:])  # [C,H,W] -> [H,W]
    homography = sample_homography(image_shape, **config['params'])
    
    # Convert image to tensor if it's not already
    if not isinstance(data['image'], torch.Tensor):
        image = torch.from_numpy(data['image']).float()
    else:
        image = data['image']
    
    # Add batch dimension if needed
    if image.dim() == 2:
        image = image.unsqueeze(0).unsqueeze(0)  # [H,W] -> [1,1,H,W]
    elif image.dim() == 3:
        image = image.unsqueeze(0)  # [C,H,W] -> [1,C,H,W]
    
    # Convert homography to 3x3 matrix format and add batch dimension
    homography_mat = flat2mat(homography.unsqueeze(0))  # [1,3,3]
    
    # Apply homography to image
    warped_image = kgm.warp_perspective(image, homography_mat, 
                                      tuple(image_shape.tolist()),
                                      mode='bilinear')
    warped_image = warped_image.squeeze(0)  # Remove batch dimension
    
    valid_mask = compute_valid_mask(image_shape, homography,
                                  config['valid_border_margin'])

    # 对关键点使用原始变换矩阵
    warped_points = warp_points(data['keypoints'], homography)
    valid_points = filter_points(warped_points, image_shape)
    warped_points = warped_points[valid_points]  # 只保留有效的关键点

    ret = {**data, 'image': warped_image, 'keypoints': warped_points,
           'valid_mask': valid_mask}
    if add_homography:
        ret['homography'] = homography
    return ret


def add_dummy_valid_mask(data):
    """Add a dummy valid mask of ones.
    
    Args:
        data: Dictionary containing the image.
        
    Returns:
        Updated data dictionary with valid mask.
    """
    if isinstance(data['image'], torch.Tensor):
        valid_mask = torch.ones(data['image'].shape[1:], dtype=torch.int32)  # [C,H,W] -> [H,W]
    else:
        valid_mask = np.ones(data['image'].shape[1:], dtype=np.int32)
    return {**data, 'valid_mask': valid_mask}


def add_keypoint_map(data):
    """Create a binary map of keypoint locations.
    
    Args:
        data: Dictionary containing the image and keypoints.
        
    Returns:
        Updated data dictionary with keypoint map.
    """
    if isinstance(data['image'], torch.Tensor):
        image_shape = data['image'].shape[1:]  # [C,H,W] -> [H,W]
        if len(data['keypoints']) > 0:  # 确保有关键点
            # 将关键点坐标转换为整数索引
            kp = torch.minimum(data['keypoints'].round().long(), 
                             torch.tensor(image_shape, device=data['keypoints'].device)-1)
            # 创建空的关键点图
            kmap = torch.zeros(image_shape, dtype=torch.int32, device=data['keypoints'].device)
            # 使用scatter将1填充到关键点位置
            indices = kp.t().long()  # 转置为[2, N]格式，用于scatter
            kmap.index_put_((indices[0], indices[1]), 
                          torch.ones(indices.shape[1], dtype=torch.int32, device=data['keypoints'].device))
        else:
            kmap = torch.zeros(image_shape, dtype=torch.int32, device=data['image'].device)
    else:
        image_shape = data['image'].shape[1:]
        if len(data['keypoints']) > 0:  # 确保有关键点
            kp = np.minimum(np.round(data['keypoints']).astype(np.int32), 
                          np.array(image_shape)-1)
            kmap = np.zeros(image_shape, dtype=np.int32)
            kmap[kp[:, 0], kp[:, 1]] = 1
        else:
            kmap = np.zeros(image_shape, dtype=np.int32)
    return {**data, 'keypoint_map': kmap}


def downsample(image, coordinates, **config):
    """Blur and downsample the image, and adjust keypoint coordinates.
    
    Args:
        image: Input image.
        coordinates: Keypoint coordinates.
        config: Configuration dictionary with blur_size and resize parameters.
        
    Returns:
        Tuple of (downsampled_image, adjusted_coordinates).
    """
    # Convert to tensor if needed
    if not isinstance(image, torch.Tensor):
        image = torch.from_numpy(image).float()
    if not isinstance(coordinates, torch.Tensor):
        coordinates = torch.from_numpy(coordinates).float()
    
    # Add batch dimension if needed
    if image.dim() == 2:
        image = image.unsqueeze(0).unsqueeze(0)  # [H,W] -> [1,1,H,W]
    elif image.dim() == 3:
        image = image.unsqueeze(0)  # [C,H,W] -> [1,C,H,W]

    # Create Gaussian kernel
    k_size = config['blur_size']
    kernel = cv.getGaussianKernel(k_size, 0)[:, 0]
    kernel = np.outer(kernel, kernel).astype(np.float32)
    kernel = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0)
    
    # Apply Gaussian blur
    pad_size = int(k_size/2)
    image = F.pad(image, [pad_size]*4, mode='reflect')
    image = F.conv2d(image, kernel)
    
    # Compute resize ratio and adjust coordinates
    ratio = torch.tensor(config['resize'], dtype=torch.float32) / torch.tensor(image.shape[2:])
    coordinates = coordinates * ratio
    
    # Resize image
    image = F.resize(image, config['resize'], antialias=True)
    
    # Remove batch dimension
    image = image.squeeze(0)  # [1,C,H,W] -> [C,H,W]
    
    return image, coordinates


def ratio_preserving_resize(image, **config):
    """Resize image while preserving aspect ratio.
    
    Args:
        image: Input image.
        config: Configuration dictionary with resize parameter.
        
    Returns:
        Resized image.
    """
    # Convert to tensor if needed
    if not isinstance(image, torch.Tensor):
        image = torch.from_numpy(image).float()
    
    # Add batch and channel dimensions if needed
    if image.dim() == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.dim() == 3:
        image = image.unsqueeze(0)
    
    target_size = torch.tensor(config['resize'])
    scales = target_size.float() / torch.tensor(image.shape[2:])
    new_size = torch.tensor(image.shape[2:]).float() * scales.max()
    
    # Resize to new size
    image = F.resize(image, tuple(new_size.int().tolist()), antialias=True)
    
    # Center crop or pad to target size
    image = F.center_crop(image, tuple(target_size.tolist()))
    
    # Remove extra dimensions
    image = image.squeeze(0)
    
    return image 