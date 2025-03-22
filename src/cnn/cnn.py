import torch 
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self, conv_config, pool_type, dropout_rate, flip_kernel, sparsity=0.0,
                use_tiled=False, use_unshared=False, use_locally_connected=False, padding_style='same'):
        super(CustomCNN, self).__init__()
        self.layers = nn.ModuleList()
        self.flip_kernel = flip_kernel
        self.sparsity = sparsity
        self.use_tiled = use_tiled
        self.use_unshared = use_unshared
        self.use_locally_connected = use_locally_connected
        self.padding_style = padding_style

        input_channels = 3
        for out_channels, kernel_size, stride, padding in conv_config:
            if self.use_tiled:
                self.layers.append(self.create_tiled_conv(input_channels, out_channels, kernel_size, stride, padding))
            elif self.use_unshared:
                self.layers.append(self.create_unshared_conv(input_channels, out_channels, kernel_size))
            elif self.use_locally_connected:
                self.layers.append(self.create_locally_connected(input_channels, out_channels, kernel_size, stride, padding))
            else:
                self.layers.append(nn.Conv2d(input_channels, out_channels, kernel_size, stride, 0))  
            input_channels = out_channels

        self.pool_type = pool_type
        flattened_size = self.get_flattened_size((3, 32, 32))  
        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(dropout_rate)

    def compute_padding(self, kernel_size, stride, input_size, padding_style):
        if padding_style == 'valid':
            return 0
        elif padding_style == 'same':
            return ((input_size - 1) * stride + kernel_size - input_size) // 2
        elif padding_style == 'full':
            return kernel_size - 1
        else:
            raise ValueError(f"Unknown padding style: {padding_style}")

    def create_tiled_conv(self, in_channels, out_channels, kernel_size, stride, padding):
        # Custom tiled convolution implementation
        return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)  
    def create_unshared_conv(self, in_channels, out_channels, kernel_size):
        # Custom unshared convolution using nn.Unfold and nn.Linear
        return nn.Sequential(
            nn.Unfold(kernel_size),
            nn.Linear(kernel_size * kernel_size * in_channels, out_channels)
        )

    def create_locally_connected(self, in_channels, out_channels, kernel_size, stride, padding):
        # Custom locally connected layer implementation
        height, width = 32, 32  
        output_height = (height + 2 * padding - kernel_size) // stride + 1
        output_width = (width + 2 * padding - kernel_size) // stride + 1
        return nn.ModuleList([
            nn.Linear(kernel_size * kernel_size * in_channels, out_channels)
            for _ in range(output_height * output_width)
        ])

    def get_flattened_size(self, input_shape):
        x = torch.zeros(1, *input_shape)
        with torch.no_grad():
            for conv in self.layers:
                x = F.relu(self.custom_conv(conv, x))
                x = self.pool_type(x)
        return x.numel()

    def custom_conv(self, conv, x):
        if isinstance(conv, nn.ModuleList):  
            output = []
            patches = F.unfold(x, kernel_size=3, stride=1).transpose(1, 2)  
            for i, layer in enumerate(conv):
                output.append(layer(patches[:, i]))
            return torch.stack(output, dim=1).view(x.size(0), -1, x.size(2) - 2, x.size(3) - 2)
        elif isinstance(conv, nn.Sequential):  
            return conv(x)
        else: 
            kernel_size = conv.kernel_size[0]
            stride = conv.stride[0]
            input_size = x.shape[-1]
            padding = self.compute_padding(kernel_size, stride, input_size, self.padding_style)

            if self.flip_kernel:
                return F.conv2d(x, conv.weight, bias=conv.bias, stride=conv.stride, padding=padding)
            else:
                
                weight = conv.weight
                flipped_weight = torch.flip(weight, dims=[2, 3])  
                return F.conv2d(x, flipped_weight, bias=conv.bias, stride=conv.stride, padding=padding)

    def apply_sparsity(self, weight):
        if self.sparsity > 0.0:
            mask = torch.rand_like(weight) > self.sparsity
            weight = weight * mask.float()  
        return weight

    def forward(self, x):
        for conv in self.layers:
            if hasattr(conv, 'weight'):
                conv.weight.data = self.apply_sparsity(conv.weight.data)
            x = F.relu(self.custom_conv(conv, x))
            x = self.pool_type(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
