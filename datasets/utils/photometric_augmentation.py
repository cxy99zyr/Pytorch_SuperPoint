import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


augmentations = [
    'additive_gaussian_noise',
    'additive_speckle_noise',
    'random_brightness',
    'random_contrast',
    'additive_shade',
    'motion_blur'
]


def additive_gaussian_noise(image, stddev_range=[5, 95]):
    """Add Gaussian noise to the image.
    
    Args:
        image: Input image (numpy array or torch tensor).
        stddev_range: Range of standard deviation for the noise.
        
    Returns:
        Noisy image.
    """
    # Convert to tensor if needed
    if not isinstance(image, torch.Tensor):
        image = torch.from_numpy(image).float()
    
    stddev = torch.empty(1).uniform_(*stddev_range)
    noise = torch.randn_like(image) * stddev
    noisy_image = torch.clamp(image + noise, 0, 255)
    
    return noisy_image


def additive_speckle_noise(image, prob_range=[0.0, 0.005]):
    """Add speckle (salt and pepper) noise to the image.
    
    Args:
        image: Input image (numpy array or torch tensor).
        prob_range: Range of probability for noise.
        
    Returns:
        Noisy image.
    """
    # Convert to tensor if needed
    if not isinstance(image, torch.Tensor):
        image = torch.from_numpy(image).float()
    
    prob = torch.empty(1).uniform_(*prob_range).item()
    sample = torch.rand_like(image)
    
    # Create noise mask
    black_mask = sample <= prob
    white_mask = sample >= (1.0 - prob)
    
    # Apply noise
    noisy_image = image.clone()
    noisy_image[black_mask] = 0.0
    noisy_image[white_mask] = 255.0
    
    return noisy_image


def random_brightness(image, max_abs_change=50):
    """Randomly adjust image brightness.
    
    Args:
        image: Input image (numpy array or torch tensor).
        max_abs_change: Maximum absolute change in brightness.
        
    Returns:
        Brightness adjusted image.
    """
    # Convert to tensor if needed
    if not isinstance(image, torch.Tensor):
        image = torch.from_numpy(image).float()
    
    delta = torch.empty(1).uniform_(-max_abs_change, max_abs_change).item()
    return torch.clamp(image + delta, 0, 255)


def random_contrast(image, strength_range=[0.5, 1.5]):
    """Randomly adjust image contrast.
    
    Args:
        image: Input image (numpy array or torch tensor).
        strength_range: Range of contrast adjustment factor.
        
    Returns:
        Contrast adjusted image.
    """
    # Convert to tensor if needed
    if not isinstance(image, torch.Tensor):
        image = torch.from_numpy(image).float()
    
    factor = torch.empty(1).uniform_(*strength_range).item()
    mean = torch.mean(image)
    return torch.clamp((image - mean) * factor + mean, 0, 255)


def additive_shade(image, nb_ellipses=20, transparency_range=[-0.5, 0.8],
                  kernel_size_range=[250, 350]):
    """Add random shading to the image.
    
    Args:
        image: Input image (numpy array or torch tensor).
        nb_ellipses: Number of ellipses to draw.
        transparency_range: Range of transparency values.
        kernel_size_range: Range of Gaussian kernel sizes.
        
    Returns:
        Shaded image.
    """
    # Convert to numpy if needed
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    
    # 确保图像是3D数组
    if len(image.shape) == 2:
        image = image[..., np.newaxis]
    
    min_dim = min(image.shape[:2]) / 4
    mask = np.zeros(image.shape[:2], np.uint8)
    
    for i in range(nb_ellipses):
        ax = int(max(np.random.rand() * min_dim, min_dim / 5))
        ay = int(max(np.random.rand() * min_dim, min_dim / 5))
        max_rad = max(ax, ay)
        x = np.random.randint(max_rad, image.shape[1] - max_rad)  # center
        y = np.random.randint(max_rad, image.shape[0] - max_rad)
        angle = np.random.rand() * 90
        cv.ellipse(mask, (x, y), (ax, ay), angle, 0, 360, 255, -1)

    transparency = np.random.uniform(*transparency_range)
    kernel_size = np.random.randint(*kernel_size_range)
    if (kernel_size % 2) == 0:  # kernel_size has to be odd
        kernel_size += 1
    
    mask = cv.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)
    mask = mask[..., np.newaxis]  # 确保mask是3D
    shaded = image * (1 - transparency * mask/255.)
    shaded = np.clip(shaded, 0, 255)
    
    # Convert back to tensor if input was tensor
    if isinstance(image, torch.Tensor):
        shaded = torch.from_numpy(shaded)
    
    # 如果输入是2D，输出也应该是2D
    if len(shaded.shape) == 3 and shaded.shape[-1] == 1:
        shaded = np.squeeze(shaded, axis=-1)
    
    return shaded


def motion_blur(image, max_kernel_size=10):
    """Apply motion blur to the image.
    
    Args:
        image: Input image (numpy array or torch tensor).
        max_kernel_size: Maximum size of the blur kernel.
        
    Returns:
        Blurred image.
    """
    # Convert to numpy if needed
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    
    # Either vertical, horizontal or diagonal blur
    mode = np.random.choice(['h', 'v', 'diag_down', 'diag_up'])
    ksize = np.random.randint(0, (max_kernel_size+1)/2)*2 + 1  # make sure is odd
    center = int((ksize-1)/2)
    kernel = np.zeros((ksize, ksize))
    
    if mode == 'h':
        kernel[center, :] = 1.
    elif mode == 'v':
        kernel[:, center] = 1.
    elif mode == 'diag_down':
        kernel = np.eye(ksize)
    elif mode == 'diag_up':
        kernel = np.flip(np.eye(ksize), 0)
    
    var = ksize * ksize / 16.
    grid = np.repeat(np.arange(ksize)[:, np.newaxis], ksize, axis=-1)
    gaussian = np.exp(-(np.square(grid-center)+np.square(grid.T-center))/(2.*var))
    kernel *= gaussian
    kernel /= np.sum(kernel)
    
    blurred = cv.filter2D(image, -1, kernel)
    
    # Convert back to tensor if input was tensor
    if isinstance(image, torch.Tensor):
        blurred = torch.from_numpy(blurred)
    
    return blurred 