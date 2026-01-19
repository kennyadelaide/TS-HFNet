import torch
import torch.nn as nn
import numpy as np


def add_uniform_noise(data, low=0.01, high=0.1):
    """Add uniform noise to input tensor while preserving device and dtype"""
    noise = torch.rand_like(data) * (high - low) + low
    return data + noise


def weights_init(model, init_type='kaiming', verbose=False):
    """
    Enhanced network parameter initialization with:
    1. Better handling of different layer types
    2. Device-aware initialization
    3. Verbose logging option
    4. Support for modern initialization schemes

    Initialization strategies:
    - Conv3D layers:
      - Regular convs: Kaiming (ReLU) or Xavier
      - 1x1x1 convs (residual): Xavier + small bias
    - BatchNorm3D: weight=1, bias=0
    - Linear layers: Same as Conv3D
    - Attention layers: Special handling

    Args:
        model (nn.Module): Model to initialize
        init_type (str): 'kaiming' or 'xavier'
        verbose (bool): Print initialization details
    """

    def _init_weights(m):
        # if verbose:
        #     print(f"Initializing layer: {m.__class__.__name__}")

        # Conv3D layers
        if isinstance(m, nn.Conv3d):
            if m.kernel_size == (1, 1, 1):  # Residual connection
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('leaky_relu', 0.1))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)
            else:  # Regular convolution
                if init_type == 'kaiming':
                    nn.init.kaiming_normal_(
                        m.weight,
                        mode='fan_out',
                        nonlinearity='leaky_relu',
                        a=0.1  # Matches your LeakyReLU slope
                    )
                else:
                    nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('leaky_relu', 0.1))

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # BatchNorm layers
        elif isinstance(m, nn.BatchNorm3d):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            # if verbose:
            #     print(f"Initialized {m.__class__.__name__} with weight=1, bias=0")

        # Linear layers (for attention modules)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('leaky_relu', 0.1))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        # Special handling for attention weights
        elif hasattr(m, 'weight_net') and isinstance(m.weight_net[-1], nn.Softmax):
            # Initialize attention weights to near-uniform
            for layer in m.weight_net:
                if isinstance(layer, nn.Conv3d):
                    nn.init.constant_(layer.weight, 0)
                    nn.init.constant_(layer.bias, 0)
            # if verbose:
            #     print(f"Initialized attention weights to uniform")

    # Apply initialization
    model.apply(_init_weights)

    # Special output layer initialization
    for name, m in model.named_modules():
        if 'out_conv' in name and isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            # if verbose:
            #     print(f"Special initialization for output layer {name}")



# weights_init(model, init_type='kaiming', verbose=True)