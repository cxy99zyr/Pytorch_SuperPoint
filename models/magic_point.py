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
        
        # 可视化预测结果
        if config.get('visualize_predictions', False) and not config['training']:
            self._visualize_predictions(image, outputs, inputs)
        
        return outputs

    def _visualize_predictions(self, image, outputs, inputs):
        """可视化预测结果。
        
        Args:
            image: 输入图像 [B,C,H,W]。
            outputs: 模型输出字典。
            inputs: 输入字典。
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 只可视化批次中的第一个样本
        img = image[0, 0].cpu().numpy()
        pred = outputs['pred'][0].cpu().numpy()
        gt_map = inputs['keypoint_map'][0].cpu().numpy() if 'keypoint_map' in inputs else None
        
        plt.figure(figsize=(15, 5))
        
        # 显示原图和预测的关键点
        plt.subplot(131)
        plt.imshow(img, cmap='gray')
        points = np.nonzero(pred)
        plt.plot(points[1], points[0], 'r.', label='Predicted')
        if 'keypoints' in inputs:
            gt_points = inputs['keypoints'][0].cpu().numpy()
            plt.plot(gt_points[:, 0], gt_points[:, 1], 'g.', label='Ground Truth')
        plt.title('Image with Keypoints')
        plt.legend()
        
        # 显示预测的概率图
        plt.subplot(132)
        plt.imshow(outputs['prob'][0].cpu().numpy(), cmap='jet')
        plt.title('Prediction Probability')
        
        # 显示真实的关键点图
        if gt_map is not None:
            plt.subplot(133)
            plt.imshow(gt_map, cmap='jet')
            plt.title('Ground Truth Map')
        
        plt.tight_layout()
        plt.savefig(f'prediction_{self.global_step if hasattr(self, "global_step") else "test"}.png')
        plt.close()

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
            logits = outputs['logits'].permute(0, 2, 3, 1)  # [B, H, W, C]
        else:
            logits = outputs['logits']
            
        # 准备标签
        keypoint_map = inputs['keypoint_map']  # [B, H, W]
        
        # 确保keypoint_map是channels_first格式
        if keypoint_map.dim() == 3:
            keypoint_map = keypoint_map.unsqueeze(1)  # [B, 1, H, W]
        
        # 使用pixel_unshuffle降采样
        labels = F.pixel_unshuffle(keypoint_map, config['grid_size'])  # [B, grid_size^2, H/grid_size, W/grid_size]
        
        # 添加dustbin通道
        B, C, H, W = labels.shape
        dustbin = torch.ones((B, 1, H, W), device=labels.device)
        labels = torch.cat([2*labels, dustbin], dim=1)  # [B, grid_size^2+1, H/grid_size, W/grid_size]
        
        # 添加随机打破平局
        rand_break = torch.rand_like(labels) * 0.1
        labels = labels + rand_break
        
        # 转换为目标格式
        labels = labels.permute(0, 2, 3, 1)  # [B, H/grid_size, W/grid_size, grid_size^2+1]
        labels = torch.argmax(labels, dim=-1)  # [B, H/grid_size, W/grid_size]
        
        # 处理有效掩码
        valid_mask = inputs.get('valid_mask', 
                              torch.ones_like(keypoint_map))
        if valid_mask.dim() == 3:
            valid_mask = valid_mask.unsqueeze(1)  # [B, 1, H, W]
            
        # 将valid_mask降采样到与labels相同的尺度
        valid_mask = F.avg_pool2d(valid_mask.float(), 
                                 kernel_size=config['grid_size'],
                                 stride=config['grid_size'])  # [B, 1, H/grid_size, W/grid_size]
        valid_mask = valid_mask.squeeze(1)  # [B, H/grid_size, W/grid_size]
        
        # 计算交叉熵损失
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), 
                             labels.reshape(-1),
                             reduction='none')  # [(B*H/grid_size*W/grid_size)]
        
        # 应用有效掩码
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
        if prob.dim() == 3:  # 如果输入是批次数据
            batch_size = prob.size(0)
            prob_nms = torch.zeros_like(prob)
            for b in range(batch_size):
                prob_nms[b] = MagicPoint._box_nms_single(
                    prob[b], size, iou, min_prob, keep_top_k)
            return prob_nms
        else:
            return MagicPoint._box_nms_single(prob, size, iou, min_prob, keep_top_k)
    
    @staticmethod
    def _box_nms_single(prob, size, iou=0.1, min_prob=0.01, keep_top_k=0):
        """对单个样本执行非最大抑制。"""
        # 转换为边界框格式
        pts = torch.nonzero(prob >= min_prob, as_tuple=False).float()
        if pts.shape[0] == 0:
            return torch.zeros_like(prob)
            
        size = size / 2.
        boxes = torch.cat([pts - size, pts + size], dim=1)  # [N,4] format: [y1,x1,y2,x2]
        scores = prob[pts[:, 0].long(), pts[:, 1].long()]
        
        # 按分数降序排序
        scores, order = scores.sort(descending=True)
        boxes = boxes[order]
        pts = pts[order]
        
        # 执行NMS
        keep = []
        while boxes.shape[0] > 0:
            keep.append(0)  # 保留当前最高分数的框
            
            if boxes.shape[0] == 1:
                break
                
            # 计算IoU
            box = boxes[0]
            others = boxes[1:]
            
            # 计算交集
            yy1 = torch.maximum(box[0], others[:, 0])
            xx1 = torch.maximum(box[1], others[:, 1])
            yy2 = torch.minimum(box[2], others[:, 2])
            xx2 = torch.minimum(box[3], others[:, 3])
            
            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            inter = w * h
            
            # 计算IoU
            area1 = (box[2] - box[0]) * (box[3] - box[1])
            area2 = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
            union = area1 + area2 - inter
            iou_scores = inter / union
            
            # 保留IoU小于阈值的框
            mask = iou_scores <= iou
            if not mask.any():
                break
                
            # 更新boxes和相关数据
            boxes = boxes[1:][mask]
            pts = pts[1:][mask]
            scores = scores[1:][mask]
        
        # 转换keep列表为实际的索引
        keep = torch.tensor(keep, device=prob.device)
        
        # 保留top-k
        if keep_top_k > 0:
            k = min(keep_top_k, len(keep))
            keep = keep[:k]
        
        # 重建概率图
        prob_nms = torch.zeros_like(prob)
        selected_pts = pts[keep]
        selected_scores = scores[keep]
        prob_nms[selected_pts[:, 0].long(), selected_pts[:, 1].long()] = selected_scores
        
        return prob_nms 