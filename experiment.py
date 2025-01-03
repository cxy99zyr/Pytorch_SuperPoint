import os
import logging
import yaml
import argparse
from pathlib import Path
from contextlib import contextmanager
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pprint import pformat as pprint

from superpoint.datasets import get_dataset
from superpoint.models import get_model
from superpoint.utils.stdout_capturing import capture_outputs
from superpoint.settings import EXPER_PATH

# 设置日志
logging.basicConfig(
    format='[%(asctime)s %(levelname)s] %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def get_module(module_name, module_cls):
    """获取指定模块的类。
    
    Args:
        module_name: 模块名称。
        module_cls: 类名称。
    
    Returns:
        模块类。
    """
    if module_name == 'datasets':
        return get_dataset(module_cls)
    elif module_name == 'models':
        return get_model(module_cls)
    else:
        raise ValueError(f'Unknown module name: {module_name}')

@contextmanager
def _init_graph(config):
    """初始化计算图。
    
    Args:
        config: 配置字典。
    
    Yields:
        元组 (model, train_loader, val_loader)。
    """
    # 设置默认训练参数
    train_config = {
        'batch_size': 32,
        'num_workers': 4
    }
    # 更新配置
    if 'train' in config:
        train_config.update(config['train'])
    config['train'] = train_config
    
    # 创建数据集
    base_dataset = get_module('datasets', config['data']['name'])(**config['data'])
    train_dataset = base_dataset.get_train_set()
    val_dataset = base_dataset.get_val_set()
    
    train_loader = DataLoader(
        train_dataset, batch_size=train_config['batch_size'],
        shuffle=True, pin_memory=True,
        num_workers=train_config['num_workers'],
        collate_fn=train_dataset.get_collate_fn())
    
    val_loader = DataLoader(
        val_dataset, batch_size=train_config['batch_size'],
        shuffle=False, pin_memory=True,
        num_workers=train_config['num_workers'],
        collate_fn=val_dataset.get_collate_fn())
    
    # 创建模型
    model = get_module('models', config['model']['name'])(**config['model'])
    
    try:
        yield model, train_loader, val_loader
    finally:
        pass  # 清理资源（如果需要）

def train(config, output_dir, pretrained_path=None):
    """训练模型。
    
    Args:
        config: 配置字典。
        output_dir: 输出目录路径。
        pretrained_path: 预训练模型路径。
    """
    logger.info("初始化...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(output_dir)
    
    # 设置设备
    device = torch.device(config['model'].get('device', 'cuda')
                         if torch.cuda.is_available() else 'cpu')
    
    # 初始化模型和数据加载器
    with _init_graph(config) as (model, train_loader, val_loader):
        model = model.to(device)
        
        # 加载预训练模型
        if pretrained_path is not None:
            state_dict = torch.load(pretrained_path, map_location=device)
            model.load_state_dict(state_dict['model_state_dict'])
            logging.info(f'Loaded pretrained model from {pretrained_path}')
        
        # 设置优化器
        optimizer = _get_optimizer(model, config['model'])
        scheduler = _get_scheduler(optimizer, config['model'])
        
        # 训练循环
        num_epochs = config['train'].get('num_epochs', 1)
        eval_interval = config['train'].get('eval_interval', 1000)
        save_interval = config['train'].get('save_interval', 5000)
        global_step = 0
        best_val_loss = float('inf')
        
        try:
            for epoch in range(num_epochs):
                model.train()
                pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
                
                for batch in pbar:
                    # 将数据移到设备上
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}
                    
                    # 前向传播和损失计算
                    outputs = model(batch, mode='train')
                    loss = model.loss(outputs, batch)
                    
                    # 反向传播和优化
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # 更新进度条
                    global_step += 1
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                    writer.add_scalar('train/loss', loss.item(), global_step)
                    
                    # 验证
                    if global_step % eval_interval == 0:
                        val_metrics = evaluate(model, val_loader, device)
                        model.train()
                        
                        # 记录验证指标
                        for name, value in val_metrics.items():
                            writer.add_scalar(f'val/{name}', value, global_step)
                        
                        # 保存最佳模型
                        val_loss = val_metrics.get('loss', float('inf'))
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            save_path = os.path.join(output_dir, 'best_model.pth')
                            torch.save({
                                'epoch': epoch,
                                'global_step': global_step,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                                'config': config,
                                'val_metrics': val_metrics
                            }, save_path)
                    
                    # 定期保存检查点
                    if save_interval and global_step % save_interval == 0:
                        save_path = os.path.join(output_dir, f'model_step{global_step}.pth')
                        torch.save({
                            'epoch': epoch,
                            'global_step': global_step,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                            'config': config
                        }, save_path)
                
                # 更新学习率
                if scheduler is not None:
                    scheduler.step()
                    writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)
        
        except KeyboardInterrupt:
            logging.info('Got Keyboard Interrupt, saving model and closing.')
        
        # 保存最终模型
        save_path = os.path.join(output_dir, 'final_model.pth')
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'config': config
        }, save_path)
        
        writer.close()


def evaluate(model, data_loader, device, max_iterations=None):
    """评估模型。
    
    参数:
        model: 要评估的模型。
        data_loader: 数据加载器。
        device: 运行设备。
        max_iterations: 最大评估迭代次数。
        
    返回:
        包含评估指标的字典。
    """
    model.eval()
    metrics = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, desc='Evaluating')):
            if max_iterations and i >= max_iterations:
                break
                
            # 将数据移到设备上
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            # 前向传播
            outputs = model(batch, mode='eval')
            batch_metrics = model.metrics(outputs, batch)
            metrics.append(batch_metrics)
    
    # 计算平均指标
    metrics_mean = {}
    for key in metrics[0].keys():
        values = [m[key].item() if isinstance(m[key], torch.Tensor) else m[key]
                 for m in metrics]
        metrics_mean[key] = np.mean(values)
    
    return metrics_mean


def predict(model, data_loader, device, keys='*'):
    """使用模型进行预测。
    
    参数:
        model: 要使用的模型。
        data_loader: 数据加载器。
        device: 运行设备。
        keys: 要预测的键。
        
    返回:
        预测结果列表。
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Predicting'):
            # 将数据移到设备上
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            # 前向传播
            outputs = model(batch, mode='pred')
            if keys == '*':
                pred = outputs
            elif isinstance(keys, str):
                pred = outputs[keys]
            else:
                pred = {k: outputs[k] for k in keys}
            
            predictions.append(pred)
    
    return predictions


def _get_optimizer(model, config):
    """创建优化器。
    
    参数:
        model: 要优化的模型。
        config: 配置字典。
        
    返回:
        优化器实例。
    """
    optimizer_config = config.get('optimizer', {})
    optimizer_name = optimizer_config.get('name', 'adam').lower()
    
    if optimizer_name == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=optimizer_config.get('weight_decay', 0.0)
        )
    else:
        raise ValueError(f'Unsupported optimizer: {optimizer_name}')


def _get_scheduler(optimizer, config):
    """创建学习率调度器。
    
    参数:
        optimizer: 优化器实例。
        config: 配置字典。
        
    返回:
        学习率调度器实例。
    """
    scheduler_config = config.get('scheduler')
    if not scheduler_config:
        return None
    
    scheduler_name = scheduler_config['name'].lower()
    if scheduler_name == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config['step_size'],
            gamma=scheduler_config['gamma']
        )
    else:
        raise ValueError(f'Unsupported scheduler: {scheduler_name}')


def _cli_train(config, output_dir, args):
    """处理训练命令行接口。
    
    参数:
        config: 配置字典。
        output_dir: 输出目录路径。
        args: 命令行参数。
    """
    # 保存配置
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    # 获取预训练模型路径
    if args.pretrained_model is not None:
        pretrained_path = os.path.join(EXPER_PATH, args.pretrained_model, 'final_model.pth')
        if not os.path.exists(pretrained_path):
            raise ValueError(f'Missing pretrained model: {pretrained_path}')
    else:
        pretrained_path = None
    
    # 训练模型
    train(config, output_dir, pretrained_path)
    
    # 评估模型
    if args.eval:
        _cli_eval(config, output_dir, args)


def _cli_eval(config, output_dir, args):
    """处理评估命令行接口。
    
    参数:
        config: 配置字典。
        output_dir: 输出目录路径。
        args: 命令行参数。
    """
    # 加载模型配置
    with open(os.path.join(output_dir, 'config.yml'), 'r') as f:
        model_config = yaml.safe_load(f)['model']
    model_config.update(config.get('model', {}))
    config['model'] = model_config
    
    # 设置设备
    device = torch.device(config['model'].get('device', 'cuda')
                         if torch.cuda.is_available() else 'cpu')
    
    # 初始化模型和数据加载器
    with _init_graph(config) as (model, _, val_loader):
        model = model.to(device)
        
        # 加载模型权重
        checkpoint = torch.load(os.path.join(output_dir, 'final_model.pth'),
                              map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 评估模型
        results = evaluate(model, val_loader, device,
                         max_iterations=config.get('eval_iter'))
    
    # 打印和导出结果
    logging.info('Evaluation results: \n{}'.format(
        pprint(results, indent=2, default=str)))
    with open(os.path.join(output_dir, 'eval.txt'), 'a') as f:
        f.write('Evaluation for {} dataset:\n'.format(config['data']['name']))
        for r, v in results.items():
            f.write('\t{}:\n\t\t{}\n'.format(r, v))
        f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    
    # 训练命令
    p_train = subparsers.add_parser('train')
    p_train.add_argument('config', type=str)
    p_train.add_argument('exper_name', type=str)
    p_train.add_argument('--eval', action='store_true')
    p_train.add_argument('--pretrained_model', type=str, default=None)
    p_train.set_defaults(func=_cli_train)
    
    # 评估命令
    p_eval = subparsers.add_parser('evaluate')
    p_eval.add_argument('config', type=str)
    p_eval.add_argument('exper_name', type=str)
    p_eval.set_defaults(func=_cli_eval)
    
    args = parser.parse_args()
    print(args.config)
    print(os.path.abspath(args.config))
    if os.access(args.config,os.R_OK):
        print('可读')
    else:
        print('不可读')
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    output_dir = os.path.join(EXPER_PATH, args.exper_name)
    
    with capture_outputs(os.path.join(output_dir, 'log')):
        logging.info('Running command {}'.format(args.command.upper()))
        args.func(config, output_dir, args) 