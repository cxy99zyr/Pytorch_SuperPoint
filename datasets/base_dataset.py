from abc import ABCMeta, abstractmethod
import torch
from torch.utils.data import Dataset

from superpoint.utils.tools import dict_update


class BaseDataset(Dataset, metaclass=ABCMeta):
    """Base dataset class.

    Arguments:
        config: A dictionary containing the configuration parameters.

    Datasets should inherit from this class and implement the following methods:
        `_init_dataset` and `__getitem__`.
    Additionally, the following static attributes should be defined:
        default_config: A dictionary of potential default configuration values (e.g. the
            size of the validation set).
    """
    split_names = ['training', 'validation', 'test']

    def get_train_set(self):
        """获取训练集。

        返回:
            训练集数据集对象。
        """
        return type(self)(split='training', **self.config)

    def get_val_set(self):
        """获取验证集。

        返回:
            验证集数据集对象。
        """
        return type(self)(split='validation', **self.config)

    def get_test_set(self):
        """获取测试集。

        返回:
            测试集数据集对象。
        """
        return type(self)(split='test', **self.config)

    @abstractmethod
    def _init_dataset(self, **config):
        """Prepare the dataset for reading.

        This method should configure the dataset for later fetching through `__getitem__`,
        such as downloading the data if it is not stored locally, or reading the list of
        data files from disk. Ideally, especially in the case of large images, this
        method should NOT read all the dataset into memory, but rather prepare for faster
        subsequent fetching.

        Arguments:
            config: A configuration dictionary, given during the object instantiation.

        Returns:
            An object subsequently passed to `__getitem__`, e.g. a list of file paths and
            set splits.
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index):
        """Get a data sample.

        This method should return a single sample from the dataset at the given index.
        The sample should be a dictionary containing the data components.

        Arguments:
            index: The index of the sample to get.

        Returns:
            A dictionary mapping component names to the corresponding data (e.g. torch.Tensor).
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """Return the total size of the dataset."""
        raise NotImplementedError

    def __init__(self, split='training', **config):
        """Initialize the dataset.

        Arguments:
            split: The dataset split to use ('training', 'validation', or 'test').
            config: Configuration parameters.
        """
        super().__init__()
        self.split = split
        if split not in self.split_names:
            raise ValueError(f'Invalid split name: {split}')

        # Update config
        self.config = dict_update(getattr(self, 'default_config', {}), config)

        # Initialize the dataset
        self.dataset = self._init_dataset(**self.config)

    def get_data_loader(self, batch_size=1, shuffle=None, num_workers=0, pin_memory=True):
        """获取数据加载器。

        参数:
            batch_size: 每个批次的样本数。
            shuffle: 是否打乱数据。如果为None，则仅在训练时打乱。
            num_workers: 用于数据加载的子进程数。
            pin_memory: 如果为True，数据加载器会将张量复制到CUDA固定内存中。

        返回:
            PyTorch DataLoader对象。
        """
        if shuffle is None:
            shuffle = (self.split == 'training')
        
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        ) 