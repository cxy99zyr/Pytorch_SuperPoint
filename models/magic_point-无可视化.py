import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia as K

from .base_model import BaseModel, Mode
from .backbones.vgg import VGGBackbone
from .homographies import homography_adaptation
from superpoint.utils.tools import dict_update


class MagicPoint(BaseModel):
    """MagicPoint模型实现。
    
    该模型用于检测图像中的关键点，使用VGG骨干网络提取特征，
    然后通过检测头预测每个像素是否为关键点。
    """
    
    input_spec = {
        'image': {
            'shape': [None, 1, None, None],  # [B,C,H,W]
            'type': torch.float32
        }
    }
    
    required_config_keys = []
    
    default_config = {
        'data_format': 'channels_first',
        'kernel_reg': 0.,
        'grid_size': 8,
        'detection_threshold': 0.4,
        'homography_adaptation': {'num': 0},
        'nms': 0,
        'top_k': 0
    }

    def __init__(self, **config):
        # 调用父类的__init__
        super().__init__(**config)
        
        # 创建VGG骨干网络
        self.backbone = VGGBackbone(self.config)
        
        # 创建检测头
        self.detector = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1+pow(self.config['grid_size'], 2), 1),
        )
        
        if self.config.get('kernel_reg', 0.) > 0:
            self.kernel_regularizer = nn.L1Loss(reduction='sum')
        else:
            self.kernel_regularizer = None
            
        # 移动模型到设备
        self.to(self.device)
        
        # 创建优化器
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config['learning_rate'])

    def _detector_head(self, features):
        """检测头实现。
        
        参数:
            features: 特征图，形状为[B, C, H, W]。
            
        返回:
            包含logits和概率的字典。
            
        异常:
            ValueError: 如果输入特征的维度不正确。
        """
        x = self.detector(features)
        
        # 检查维度
        B, C, H, W = x.shape
        expected_channels = 1 + pow(self.config['grid_size'], 2)
        if C != expected_channels:
            raise ValueError(f'检测头输出通道数错误：期望 {expected_channels}，实际 {C}')
        
        # 计算概率
        prob = F.softmax(x, dim=1)
        # 移除"非关键点"的dustbin通道
        prob = prob[:, :-1]
        
        # 重新排列为网格点
        try:
            prob = prob.view(B, self.config['grid_size'], 
                           self.config['grid_size'], H, W)
            prob = prob.permute(0, 3, 1, 4, 2).contiguous()
            prob = prob.view(B, H*self.config['grid_size'], 
                           W*self.config['grid_size'])
        except RuntimeError as e:
            raise ValueError(f'维度重排列失败：{str(e)}。请检查grid_size配置是否正确。')
        
        return {'logits': x, 'prob': prob}

    def _model(self, inputs, mode, **config):
        """前向传播实现。
        
        参数:
            inputs: 输入字典，包含'image'键。
            mode: 运行模式（训练/评估/预测）。
            config: 额外的配置参数。
            
        返回:
            包含模型输出的字典。
        """
        config = {**self.config, **config}
        config['training'] = (mode == Mode.TRAIN)
        image = inputs['image']
        
        # 数据集已经输出channels_first格式 [B,C,H,W]，不需要转换
            
        def net(image):
            features = self.backbone(image)
            outputs = self._detector_head(features)
            return outputs
        
        if (mode == Mode.PRED) and config['homography_adaptation']['num']:
            outputs = homography_adaptation(image, net, 
                                         config['homography_adaptation'])
        else:
            outputs = net(image)
        
        prob = outputs['prob']
        if config['nms']:
            prob = self._box_nms(prob, config['nms'],
                               min_prob=config['detection_threshold'],
                               keep_top_k=config['top_k'])
            outputs['prob_nms'] = prob
            
        pred = (prob >= config['detection_threshold']).float()
        outputs['pred'] = pred
        
        return outputs

    def _loss(self, outputs, inputs, **config):
        """计算损失。
        
        参数:
            outputs: 模型输出字典。
            inputs: 输入字典。
            config: 额外的配置参数。
            
        返回:
            标量损失值。
        """
        config = {**self.config, **config}
        
        if config['data_format'] == 'channels_first':
            logits = outputs['logits'].permute(0, 2, 3, 1)
        else:
            logits = outputs['logits']
            
        # 准备标签
        keypoint_map = inputs['keypoint_map']
        labels = keypoint_map.unsqueeze(-1).float()
        labels = F.pixel_unshuffle(labels, config['grid_size'])
        shape = list(labels.shape)
        shape[-1] = 1
        labels = torch.cat([2*labels, torch.ones(*shape, device=labels.device)], -1)
        
        # 添加小的随机值以打破相等值的平局
        rand_break = torch.rand_like(labels) * 0.1
        labels = torch.argmax(labels + rand_break, dim=-1)
        
        # 处理有效码
        valid_mask = inputs.get('valid_mask', 
                              torch.ones_like(keypoint_map))
        valid_mask = valid_mask.unsqueeze(-1).float()
        valid_mask = F.pixel_unshuffle(valid_mask, config['grid_size'])
        valid_mask = valid_mask.prod(dim=-1)  # 在通道维度上进行AND操作
        
        # 计算交叉熵损失
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), 
                             labels.reshape(-1),
                             reduction='none')
        loss = (loss * valid_mask.reshape(-1)).mean()
        
        # 添加L1正则化（如果配置）
        if self.kernel_regularizer is not None:
            for param in self.parameters():
                loss = loss + config['kernel_reg'] * torch.sum(torch.abs(param))
        
        return loss

    def _metrics(self, outputs, inputs, **config):
        """计算评估指标。
        
        参数:
            outputs: 模型输出字典。
            inputs: 输入字典。
            config: 额外的配置参数。
            
        返回:
            包含指标的字典。
        """
        config = {**self.config, **config}
        
        valid_mask = inputs.get('valid_mask', 
                              torch.ones_like(inputs['keypoint_map']))
        pred = valid_mask * outputs['pred']
        labels = inputs['keypoint_map']
        
        precision = (pred * labels).sum() / (pred.sum() + 1e-8)
        recall = (pred * labels).sum() / (labels.sum() + 1e-8)
        
        return {'precision': precision, 'recall': recall}

    @staticmethod
    def _box_nms(prob, size, iou=0.1, min_prob=0.01, keep_top_k=0):
        """执行非最大抑制。
        
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
        boxes = torch.cat([pts - size, pts + size], dim=1)  # [N,4] format: [y1,x1,y2,x2]
        scores = prob[pts[:, 0].long(), pts[:, 1].long()]
        
        # 计算边界框面积
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # 按分数降序排序
        _, order = scores.sort(0, descending=True)
        
        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break
            i = order[0]
            keep.append(i.item())
            
            # 计算IoU
            yy1 = boxes[order[1:], 0].clamp(min=boxes[i, 0].item())
            xx1 = boxes[order[1:], 1].clamp(min=boxes[i, 1].item())
            yy2 = boxes[order[1:], 2].clamp(max=boxes[i, 2].item())
            xx2 = boxes[order[1:], 3].clamp(max=boxes[i, 3].item())
            
            w = (xx2 - xx1).clamp(min=0)
            h = (yy2 - yy1).clamp(min=0)
            inter = w * h
            
            ovr = inter / (area[i] + area[order[1:]] - inter)
            
            # 获取重叠小于阈值的索引
            ids = (ovr <= iou).nonzero().squeeze()
            if ids.numel() == 0:
                break
            order = order[ids + 1]
        
        keep = torch.tensor(keep, device=prob.device)
        
        # 保留top-k
        if keep_top_k > 0:
            k = min(keep_top_k, keep.shape[0])
            keep = keep[:k]
        
        # 重建概率图
        prob_nms = torch.zeros_like(prob)
        prob_nms[pts[keep, 0].long(), pts[keep, 1].long()] = scores[keep]
        
        return prob_nms 