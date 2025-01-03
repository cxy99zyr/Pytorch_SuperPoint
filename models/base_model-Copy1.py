from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os.path as osp
from typing import Dict, Any, Optional, Union
import os
from superpoint.utils.tools import dict_update


class Mode:
    """Model running mode."""
    TRAIN = 'train'
    EVAL = 'eval'
    PRED = 'pred'


class BaseModel(nn.Module, metaclass=ABCMeta):
    """Base model class.

    Args:
        data: A dictionary of datasets, can include the keys
            "training", "validation", and "test".
        device: The device to run the model on.
        data_shape: A dictionary, where the keys are the input features of the prediction
            network and the values are the associated shapes. Only required if `data` is
            empty or None.
        config: A dictionary containing the configuration parameters.
            Entries "batch_size" and "learning_rate" are required.

    Models should inherit from this class and implement the following methods:
        `_model`, `_loss`, and `_metrics`.
    Additionally, the following static attributes should be defined:
        input_spec: A dictionary, where the keys are the input features (e.g. "image")
            and the associated values are dictionaries containing "shape" (list of
            dimensions, e.g. [N, C, H, W] where None indicates an unconstrained
            dimension) and "type" (e.g. torch.float32).
        required_config_keys: A list containing the required configuration entries.
        default_config: A dictionary of potential default configuration values.
    """
    dataset_names = {'training', 'validation', 'test'}
    required_baseconfig = ['batch_size', 'learning_rate']
    _default_config = {'eval_batch_size': 1, 'pred_batch_size': 1}

    def forward(self, inputs: Dict[str, torch.Tensor], mode: str = Mode.TRAIN, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.
        
        Args:
            inputs: A dictionary of input features.
            mode: Running mode (train/eval/pred).
            **kwargs: Additional arguments passed to _model.
            
        Returns:
            Model outputs dictionary.
        """
        return self._model(inputs, mode, **{**self.config, **kwargs})
        
    @abstractmethod
    def _model(self, inputs: Dict[str, torch.Tensor], mode: str,
               **config) -> Dict[str, torch.Tensor]:
        """Implements the forward pass of the model.

        Args:
            inputs: A dictionary of input features, where the keys are their names
                (e.g. "image") and the values are tensors.
            mode: An attribute of the Mode class.
            config: A configuration dictionary.

        Returns:
            A dictionary of outputs, where the keys are their names (e.g. "logits")
            and the values are the corresponding tensors.
        """
        raise NotImplementedError

    def loss(self, outputs, inputs, **config):
        """计算损失的公开接口。
        
        参数:
            outputs: 模型输出字典。
            inputs: 输入字典。
            config: 额外的配置参数。
            
        返回:
            标量损失值。
        """
        return self._loss(outputs, inputs, **config)
        
    @abstractmethod
    def _loss(self, outputs: Dict[str, torch.Tensor],
              inputs: Dict[str, torch.Tensor], **config) -> torch.Tensor:
        """Implements the computation of the training loss.

        Args:
            outputs: A dictionary, as returned by _model called with mode=Mode.TRAIN.
            inputs: A dictionary of input features.
            config: A configuration dictionary.

        Returns:
            A tensor corresponding to the loss to be minimized during training.
        """
        raise NotImplementedError

    @abstractmethod
    def _metrics(self, outputs: Dict[str, torch.Tensor],
                inputs: Dict[str, torch.Tensor],
                **config) -> Dict[str, torch.Tensor]:
        """Implements the computation of the evaluation metrics.

        Args:
            outputs: A dictionary, as returned by _model called with mode=Mode.EVAL.
            inputs: A dictionary of input features.
            config: A configuration dictionary.

        Returns:
            A dictionary of metrics, where the keys are their names (e.g. "accuracy")
            and the values are the corresponding tensors.
        """
        raise NotImplementedError

    def __init__(self, data: Dict = {}, device: str = 'cuda',
                 data_shape: Optional[Dict] = None, **config):
        """Initialize the model.

        Args:
            data: Dictionary of datasets.
            device: Device to run the model on.
            data_shape: Dictionary of input shapes.
            **config: Configuration parameters.
        """
        super().__init__()
        self.datasets = data
        self.data_shape = data_shape
        self.device = device
        self.name = self.__class__.__name__.lower()
        
        # Update config
        self.config = dict_update(self._default_config,
                                getattr(self, 'default_config', {}))
        self.config = dict_update(self.config, config)
        
        # Check required configs
        required = self.required_baseconfig + getattr(self, 'required_config_keys', [])
        for r in required:
            assert r in self.config, f'Required configuration entry: "{r}"'
        assert set(self.datasets) <= self.dataset_names, \
            f'Unknown dataset name: {set(self.datasets)-self.dataset_names}'

    def train_step(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a training step.

        Args:
            inputs: Dictionary of input tensors.

        Returns:
            Dictionary containing the loss and any other metrics.
        """
        self.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self._model(inputs, Mode.TRAIN, **self.config)
        loss = self._loss(outputs, inputs, **self.config)
        
        # Add regularization losses
        if hasattr(self, 'reg_loss'):
            loss += self.reg_loss
            
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item()}

    def eval_step(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform an evaluation step.

        Args:
            inputs: Dictionary of input tensors.

        Returns:
            Dictionary of metrics.
        """
        self.eval()
        with torch.no_grad():
            outputs = self._model(inputs, Mode.EVAL, **self.config)
            metrics = self._metrics(outputs, inputs, **self.config)
        return metrics

    def predict(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform prediction.

        Args:
            inputs: Dictionary of input tensors.

        Returns:
            Dictionary of predictions.
        """
        self.eval()
        with torch.no_grad():
            outputs = self._model(inputs, Mode.PRED, **self.config)
        return outputs

    def train_model(self, iterations: int, validation_interval: int = 100,
                   output_dir: Optional[str] = None,
                   save_interval: Optional[int] = None,
                   checkpoint_path: Optional[str] = None,
                   max_grad_norm: float = 1.0) -> None:
        """训练模型。

        参数:
            iterations: 训练迭代次数。
            validation_interval: 验证间隔。
            output_dir: 保存检查点的目录。
            save_interval: 保存检查点的间隔。
            checkpoint_path: 加载检查点的路径。
            max_grad_norm: 梯度裁剪的最大范数。
        """
        if checkpoint_path:
            self.load(checkpoint_path)
            
        # 创建学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5,
            verbose=True, min_lr=1e-6
        )
        
        train_loader = DataLoader(
            self.datasets['training'],
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        train_iterator = iter(train_loader)
        
        best_val_loss = float('inf')
        running_loss = 0.0
        
        for i in tqdm(range(iterations), desc='训练进度'):
            # 获取下一个批次
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)
                
            # 将批次移到设备上
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 训练步骤
            metrics = self.train_step(batch)
            running_loss += metrics['loss']
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            
            # 验证
            if validation_interval and i % validation_interval == 0:
                val_metrics = self.evaluate(self.datasets['validation'])
                metrics.update({f'val_{k}': v for k, v in val_metrics.items()})
                
                # 更新学习率调度器
                scheduler.step(val_metrics.get('loss', 0))
                
                # 保存最佳模型
                if output_dir and val_metrics.get('loss', float('inf')) < best_val_loss:
                    best_val_loss = val_metrics.get('loss')
                    self.save(osp.join(output_dir, 'best_model.pth'))
                
                # 显示训练状态
                avg_loss = running_loss / validation_interval
                running_loss = 0.0
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'\n迭代 {i}/{iterations}:')
                print(f'  平均训练损失: {avg_loss:.4f}')
                print(f'  验证损失: {val_metrics.get("loss", "N/A")}')
                print(f'  当前学习率: {current_lr:.6f}')
                
            # 保存检查点
            if save_interval and output_dir and i % save_interval == 0:
                self.save(osp.join(output_dir, f'model_{i}.pth'))

    def evaluate(self, dataset, max_iterations: Optional[int] = None) -> Dict[str, float]:
        """Evaluate the model on a dataset.

        Args:
            dataset: Dataset to evaluate on.
            max_iterations: Maximum number of iterations.

        Returns:
            Dictionary of metrics.
        """
        eval_loader = DataLoader(
            dataset,
            batch_size=self.config['eval_batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        metrics_list = []
        for i, batch in enumerate(tqdm(eval_loader, desc='Evaluating')):
            if max_iterations and i >= max_iterations:
                break
                
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Evaluation step
            metrics = self.eval_step(batch)
            metrics_list.append({k: v.item() for k, v in metrics.items()})
            
        # Average metrics
        metrics_mean = {}
        for k in metrics_list[0].keys():
            metrics_mean[k] = np.mean([m[k] for m in metrics_list])
            
        return metrics_mean

    def save(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint to.
        """
        os.makedirs(osp.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)

    def load(self, path: str) -> None:
        """Load model checkpoint.

        Args:
            path: Path to load checkpoint from.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config.update(checkpoint['config']) 