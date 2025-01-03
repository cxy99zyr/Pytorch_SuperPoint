import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGBlock(nn.Module):
    """Basic VGG block with optional batch normalization."""
    def __init__(self, in_channels, out_channels, kernel_size, 
                 batch_normalization=True, kernel_reg=0., **kwargs):
        """Initialize the VGG block.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolutional kernel.
            batch_normalization: Whether to use batch normalization.
            kernel_reg: L2 regularization factor.
            **kwargs: Additional arguments for the convolutional layer.
        """
        super().__init__()
        
        # Conv layer
        padding = kernel_size // 2  # Same padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                             padding=padding, **kwargs)
        
        # Optional batch normalization
        self.batch_normalization = batch_normalization
        if batch_normalization:
            self.bn = nn.BatchNorm2d(out_channels)
            
        # L2 regularization
        self.kernel_reg = kernel_reg
        
    def forward(self, x):
        """Forward pass of the block.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor.
        """
        # Convolution
        x = self.conv(x)
        
        # Batch normalization if enabled
        if self.batch_normalization:
            x = self.bn(x)
            
        # ReLU activation
        x = F.relu(x)
        
        # Add L2 regularization loss if needed
        if self.kernel_reg > 0 and self.training:
            reg_loss = self.kernel_reg * torch.sum(torch.square(self.conv.weight))
            if hasattr(self, 'reg_loss'):
                self.reg_loss += reg_loss
            else:
                self.reg_loss = reg_loss
                
        return x


class VGGBackbone(nn.Module):
    """VGG backbone network."""
    def __init__(self, config):
        """Initialize the VGG backbone.
        
        Args:
            config: Configuration dictionary containing:
                - batch_normalization: Whether to use batch normalization
                - kernel_reg: L2 regularization factor
        """
        super().__init__()
        
        self.config = config
        params = {
            'batch_normalization': config.get('batch_normalization', True),
            'kernel_reg': config.get('kernel_reg', 0.)
        }
        
        # Build the network
        self.conv1_1 = VGGBlock(1, 64, 3, **params)
        self.conv1_2 = VGGBlock(64, 64, 3, **params)
        self.pool1 = nn.MaxPool2d(2, stride=2, padding=0)
        
        self.conv2_1 = VGGBlock(64, 64, 3, **params)
        self.conv2_2 = VGGBlock(64, 64, 3, **params)
        self.pool2 = nn.MaxPool2d(2, stride=2, padding=0)
        
        self.conv3_1 = VGGBlock(64, 128, 3, **params)
        self.conv3_2 = VGGBlock(128, 128, 3, **params)
        self.pool3 = nn.MaxPool2d(2, stride=2, padding=0)
        
        self.conv4_1 = VGGBlock(128, 256, 3, **params)
        self.conv4_2 = VGGBlock(256, 256, 3, **params)
        
    def forward(self, x):
        """Forward pass of the network.
        
        Args:
            x: Input tensor of shape [B, C, H, W].
            
        Returns:
            Output tensor.
        """
        # First block
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)
        
        # Second block
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)
        
        # Third block
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.pool3(x)
        
        # Fourth block
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        
        return x 