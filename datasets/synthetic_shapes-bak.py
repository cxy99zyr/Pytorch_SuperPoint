import numpy as np
import cv2
import os
import tarfile
from pathlib import Path
from tqdm import tqdm
import shutil
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from .base_dataset import BaseDataset
from . import synthetic_dataset
from .utils import pipeline
from .utils.pipeline import parse_primitives
from superpoint.settings import DATA_PATH


def visualize_sample(image, keypoints=None, keypoint_map=None, title=None, save_path=None):
    """可视化一个数据样本。
    
    Args:
        image: 图像张量 [C,H,W] 或 [H,W]，可以是tensor或numpy数组。
        keypoints: 关键点坐标 [N,2]，格式为(y,x)。
        keypoint_map: 关键点图 [H,W]。
        title: 图像标题。
        save_path: 保存路径。
    """
    plt.figure(figsize=(10, 5))
    
    # 显示原图
    plt.subplot(121)
    
    # 处理图像数据
    if isinstance(image, torch.Tensor):
        if image.dim() == 3:
            img = image[0].cpu().numpy()
        else:
            img = image.cpu().numpy()
    else:  # numpy array
        if image.ndim == 3:
            img = image[0]
        else:
            img = image
            
    plt.imshow(img, cmap='gray')
    
    # 处理关键点数据
    if keypoints is not None:
        if isinstance(keypoints, torch.Tensor):
            keypoints = keypoints.cpu().numpy()
        # 注意：keypoints格式为(y,x)，但plt.plot需要(x,y)顺序
        plt.plot(keypoints[:, 1], keypoints[:, 0], 'r.')
    plt.title('Image with Keypoints' if title is None else title)
    
    # 显示关键点图
    if keypoint_map is not None:
        plt.subplot(122)
        if isinstance(keypoint_map, torch.Tensor):
            keypoint_map = keypoint_map.cpu().numpy()
        plt.imshow(keypoint_map, cmap='jet')
        plt.title('Keypoint Map')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def collate_fn(batch):
    """自定义的collate函数，用于处理不同大小的张量。
    
    Args:
        batch: 包含字典的列表，每个字典包含一个样本的数据。
        
    Returns:
        合并后的批次数据字典。
    """
    # 初始化输出字典
    collated = {}
    
    # 获取所有键
    keys = batch[0].keys()
    
    for key in keys:
        if key == 'keypoints':
            # 对于关键点，我们保存为列表
            collated[key] = [sample[key] for sample in batch]
        elif key == 'keypoint_map':
            # 对于关键点图，我们可以正常stack
            collated[key] = torch.stack([sample[key] for sample in batch])
        elif key == 'valid_mask':
            # 对于有效掩码，我们可以正常stack
            collated[key] = torch.stack([sample[key] for sample in batch])
        elif key == 'image':
            # 对于图像，我们可以正常stack
            collated[key] = torch.stack([sample[key] for sample in batch])
        else:
            # 对于其他数据，我们尝试stack，如果失败则保持为列表
            try:
                collated[key] = torch.stack([sample[key] for sample in batch])
            except:
                collated[key] = [sample[key] for sample in batch]
    
    return collated


class SyntheticShapes(BaseDataset):
    default_config = {
        'primitives': 'all',
        'truncate': {},
        'validation_size': -1,
        'test_size': -1,
        'on-the-fly': False,
        'cache_in_memory': False,
        'suffix': None,
        'add_augmentation_to_test_set': False,
        'num_parallel_calls': 10,
        'generation': {
            #陈雪岩
            'split_sizes': {'training': 100, 'validation': 10, 'test': 20},
            #'split_sizes': {'training': 10000, 'validation': 200, 'test': 500},
            'image_size': [960, 1280],
            'random_seed': 0,
            'params': {
                'generate_background': {
                    'min_kernel_size': 150, 'max_kernel_size': 500,
                    'min_rad_ratio': 0.02, 'max_rad_ratio': 0.031},
                'draw_stripes': {'transform_params': (0.1, 0.1)},
                'draw_multiple_polygons': {'kernel_boundaries': (50, 100)}
            },
        },
        'preprocessing': {
            'resize': [240, 320],
            'blur_size': 11,
        },
        'augmentation': {
            'photometric': {
                'enable': False,
                'primitives': 'all',
                'params': {},
                'random_order': True,
            },
            'homographic': {
                'enable': False,
                'params': {},
                'valid_border_margin': 0,
            },
        }
    }
    drawing_primitives = [
        'draw_lines',
        'draw_polygon',
        'draw_multiple_polygons',
        'draw_ellipses',
        'draw_star',
        'draw_checkerboard',
        'draw_stripes',
        'draw_cube',
        'gaussian_noise'
    ]

    def dump_primitive_data(self, primitive, tar_path, config):
        """Generate and save synthetic data for a primitive.
        
        Args:
            primitive: Name of the drawing primitive.
            tar_path: Path to save the tar file.
            config: Configuration dictionary.
        """
        temp_dir = Path(os.environ.get('TMPDIR', '/tmp'), primitive)

        print(f'\n开始生成 {primitive} 的合成数据...')
        synthetic_dataset.set_random_state(np.random.RandomState(
            config['generation']['random_seed']))
        
        for split, size in self.config['generation']['split_sizes'].items():
            im_dir, pts_dir = [Path(temp_dir, i, split) for i in ['images', 'points']]
            im_dir.mkdir(parents=True, exist_ok=True)
            pts_dir.mkdir(parents=True, exist_ok=True)

            for i in tqdm(range(size), desc=split, leave=False):
                image = synthetic_dataset.generate_background(
                    config['generation']['image_size'],
                    **config['generation']['params']['generate_background'])
                points = np.array(getattr(synthetic_dataset, primitive)(
                    image, **config['generation']['params'].get(primitive, {})))
                points = np.flip(points, 1)  # reverse convention with opencv

                b = config['preprocessing']['blur_size']
                image = cv2.GaussianBlur(image, (b, b), 0)
                points = (points * np.array(config['preprocessing']['resize'], np.float64)
                         / np.array(config['generation']['image_size'], np.float64))
                image = cv2.resize(image, tuple(config['preprocessing']['resize'][::-1]),
                                 interpolation=cv2.INTER_LINEAR)

                cv2.imwrite(str(Path(im_dir, f'{i}.png')), image)
                np.save(Path(pts_dir, f'{i}.npy'), points)

        # Pack into a tar file
        tar = tarfile.open(tar_path, mode='w:gz')
        tar.add(temp_dir, arcname=primitive)
        tar.close()
        shutil.rmtree(temp_dir)
        print(f'Tarfile dumped to {tar_path}.')

    def _init_dataset(self, **config):
        """Initialize the dataset.
        
        Args:
            config: Configuration dictionary.
            
        Returns:
            Dictionary containing file paths for each split.
        """
        # Parse drawing primitives
        primitives = parse_primitives(config['primitives'], self.drawing_primitives)

        if config['on-the-fly']:
            return None

        basepath = Path(
            DATA_PATH, 'synthetic_shapes' +
            ('_{}'.format(config['suffix']) if config['suffix'] is not None else ''))
        basepath.mkdir(parents=True, exist_ok=True)

        splits = {s: {'images': [], 'points': []}
                 for s in ['training', 'validation', 'test']}
        for primitive in primitives:
            tar_path = Path(basepath, f'{primitive}.tar.gz')
            if not tar_path.exists():
                self.dump_primitive_data(primitive, tar_path, config)

            # Untar locally
            print(f'Extracting archive for primitive {primitive}.')
            tar = tarfile.open(tar_path)
            temp_dir = Path(os.environ.get('TMPDIR', '/tmp'))
            tar.extractall(path=temp_dir)
            tar.close()

            # Gather filenames in all splits, optionally truncate
            truncate = config['truncate'].get(primitive, 1)
            path = Path(temp_dir, primitive)
            for s in splits:
                e = [str(p) for p in Path(path, 'images', s).iterdir()]
                f = [p.replace('images', 'points') for p in e]
                f = [p.replace('.png', '.npy') for p in f]
                splits[s]['images'].extend(e[:int(truncate*len(e))])
                splits[s]['points'].extend(f[:int(truncate*len(f))])

        # Shuffle
        for s in splits:
            perm = np.random.RandomState(0).permutation(len(splits[s]['images']))
            for obj in ['images', 'points']:
                splits[s][obj] = np.array(splits[s][obj])[perm].tolist()

        # Store the file paths
        self.splits = splits
        self.size = {s: len(splits[s]['images']) for s in splits}
        return splits

    def __len__(self):
        """Return the size of the current split."""
        if self.config['on-the-fly']:
            return self.config['generation']['split_sizes'][self.split]
        return self.size[self.split]

    def _generate_shape(self):
        """生成一个合成形状。
        
        Returns:
            Tuple of (image, points)。
        """
        primitives = parse_primitives(self.config['primitives'], self.drawing_primitives)
        primitive = np.random.choice(primitives)
        
        image = synthetic_dataset.generate_background(
            self.config['generation']['image_size'],
            **self.config['generation']['params']['generate_background'])
        points = np.array(getattr(synthetic_dataset, primitive)(
            image, **self.config['generation']['params'].get(primitive, {})))
        
        # 可视化生成的形状
        if self.config.get('visualize_generation', False):
            plt.figure(figsize=(5, 5))
            plt.imshow(image, cmap='gray')
            plt.plot(points[:, 0], points[:, 1], 'r.')
            plt.title(f'Generated {primitive}')
            plt.savefig(f'generated_{primitive}.png')
            plt.close()
        
        # Convert to tensor and adjust dimensions
        image = torch.from_numpy(image).float().unsqueeze(0)  # [H,W] -> [1,H,W]
        points = torch.from_numpy(np.flip(points, 1)).float()  # reverse convention
        
        return image, points

    def __getitem__(self, index):
        """获取一个数据样本。
        
        Args:
            index: 样本索引。
            
        Returns:
            包含数据样本的字典。
        """
        if self.config['on-the-fly']:
            image, points = self._generate_shape()
            # 可视化原始生成的样本
            if self.config.get('visualize_samples', False):
                visualize_sample(image, points, title='Original Generated Sample',
                               save_path=f'sample_original_{index}.png')
            
            # Apply preprocessing
            image, points = pipeline.downsample(image, points, **self.config['preprocessing'])
        else:
            # Read image and points
            image_path = self.splits[self.split]['images'][index]
            points_path = self.splits[self.split]['points'][index]
            
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = torch.from_numpy(image).float().unsqueeze(0)  # [H,W] -> [1,H,W]
            points = torch.from_numpy(np.load(points_path)).float()
            # 可视化原始生成的样本
            if self.config.get('visualize_samples', False):
                visualize_sample(image, points, title='Original Generated Sample',
                               save_path=f'sample_original_{index}.png')
        # Create data dictionary
        data = {'image': image, 'keypoints': points}
        
        # Add valid mask
        data = pipeline.add_dummy_valid_mask(data)

        # Apply augmentation
        if self.split == 'training' or self.config['add_augmentation_to_test_set']:
            if self.config['augmentation']['photometric']['enable']:
                data = pipeline.photometric_augmentation(
                    data, **self.config['augmentation']['photometric'])
                # 可视化光度增强后的结果
                if self.config.get('visualize_augmentation', False):
                    visualize_sample(data['image'], data['keypoints'],
                                   title='After Photometric Augmentation',
                                   save_path=f'sample_photometric_{index}.png')
                    
            if self.config['augmentation']['homographic']['enable']:
                data = pipeline.homographic_augmentation(
                    data, **self.config['augmentation']['homographic'])
                # 可视化几何增强后的结果
                if self.config.get('visualize_augmentation', False):
                    visualize_sample(data['image'], data['keypoints'],
                                   title='After Homographic Augmentation',
                                   save_path=f'sample_homographic_{index}.png')

        # Add keypoint map and normalize image
        data = pipeline.add_keypoint_map(data)
        data['image'] = data['image'] / 255.
        
        # 可视化最终处理后的样本
        if self.config.get('visualize_samples', False):
            visualize_sample(data['image'], data['keypoints'], data['keypoint_map'],
                           title='Final Processed Sample',
                           save_path=f'sample_final_{index}.png')

        return data

    @staticmethod
    def get_collate_fn():
        """返回用于DataLoader的collate_fn。"""
        return collate_fn 