import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipFusionBlock(nn.Module):

    def __init__(self, in_channels, num_modals, reduction_ratio=4):
        super().__init__()
        self.num_modals = num_modals

        self.fusion = nn.Sequential(
            nn.Conv3d(in_channels*self.num_modals, in_channels*3 // reduction_ratio, 1),
            nn.InstanceNorm3d(in_channels*3 // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels*3 // reduction_ratio, in_channels, 1),
            nn.InstanceNorm3d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_list):

        concat_features = torch.cat(x_list, dim=1)
        return self.fusion(concat_features)

class UpSampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            padding=0
        )
        self.conv = DoubleConv(out_channels * 2, out_channels)


        self.spatial_att = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        x1_up = self.up(x1)

        diffZ = x2.size()[2] - x1_up.size()[2]
        diffY = x2.size()[3] - x1_up.size()[3]
        diffX = x2.size()[4] - x1_up.size()[4]

        x1_up = F.pad(x1_up, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2,
                              diffZ // 2, diffZ - diffZ // 2])

        avg_out = torch.mean(x1_up, dim=1, keepdim=True)
        max_out, _ = torch.max(x1_up, dim=1, keepdim=True)
        spatial_att = self.spatial_att(torch.cat([avg_out, max_out], dim=1))
        x1_up = x1_up * spatial_att

        x = torch.cat([x1_up, x2], dim=1)
        return self.conv(x)


class TransformerBlock(nn.Module):
    def __init__(self, in_channels, num_heads=8, dropout=0.1, hidden_dim=512):

        super().__init__()


        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)


        self.attention = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # (batch, seq, features)
        )


        self.ffn = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_channels),
            nn.Dropout(dropout)
        )


        self.position_embedding = nn.Parameter(torch.randn(1, 1, in_channels))

    def forward(self, x):

        batch, channels, d, h, w = x.shape

        x_flat = x.view(batch, channels, -1).permute(0, 2, 1)  # [batch, d*h*w, channels]

        x_flat = x_flat + self.position_embedding

        attn_output, _ = self.attention(
            query=x_flat,
            key=x_flat,
            value=x_flat,
            need_weights=False
        )
        x_flat = x_flat + attn_output
        x_flat = self.norm1(x_flat)

        ffn_output = self.ffn(x_flat)
        x_flat = x_flat + ffn_output
        x_flat = self.norm2(x_flat)

        x_out = x_flat.permute(0, 2, 1).view(batch, channels, d, h, w)

        return x_out



class SqueezeExcitation(nn.Module):

    def __init__(self, in_channels, reduction_ratio=2, dimensionality=2):
        super().__init__()
        assert dimensionality in [2, 3], "dimensionality must be 2 or 3"

        self.dimensionality = dimensionality
        hidden_dim = max(in_channels, 1)

        if dimensionality == 2:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.global_pool = nn.AdaptiveAvgPool3d(1)


        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim, bias=False),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):

        squeeze = self.global_pool(x)
        squeeze = squeeze.view(squeeze.size(0), -1)  # (B,C)

        weights = self.mlp(squeeze)

        if self.dimensionality == 2:
            weights = weights.view(-1, x.size(1), 1, 1)
        else:
            weights = weights.view(-1, x.size(1), 1, 1, 1)

        return x * weights.expand_as(x)


class CBAM3D(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):

        super(CBAM3D, self).__init__()


        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.channel_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )

        assert kernel_size in (3, 7), "kernel size is 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.spatial_conv = nn.Conv3d(
            2, 1, kernel_size=kernel_size,
            padding=padding, bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        original = x


        #  [B,C,D,H,W] -> [B,C,1,1,1] -> [B,C]
        avg_out = self.channel_fc(self.avg_pool(x).flatten(1))

        max_out = self.channel_fc(self.max_pool(x).flatten(1))

        #  [B,C,1,1,1]
        channel_weights = self.sigmoid(avg_out + max_out).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        x = x * channel_weights

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        spatial_weights = self.sigmoid(
            self.spatial_conv(torch.cat([avg_out, max_out], dim=1))
        )

        x = x * spatial_weights
        return x

class ASPP(nn.Module):


    def __init__(self, in_channels, out_channels, rates=(1, 2, 4, 8)):
        super().__init__()
        self.convs = nn.ModuleList()
        for rate in rates:
            self.convs.append(nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 3,
                          padding=rate, dilation=rate),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(0.2),
            ))
        self.global_conv = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, out_channels, 1),
            nn.LeakyReLU(0.2),
        )
        self.fusion = DoubleConv(out_channels * (len(rates) + 1), out_channels)
        self.cbam= CBAM3D(out_channels, reduction_ratio=4)

    def forward(self, x):
        features = [self.cbam(conv(x)) for conv in self.convs]
        global_feat = self.global_conv(x)
        global_feat = F.interpolate(global_feat, size=x.shape[2:],
                                    mode='trilinear', align_corners=True)
        features.append(self.cbam(global_feat))
        return self.fusion(torch.cat(features, dim=1))

class CrissCrossAttention(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv3d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv3d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, depth, height, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, depth * height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, depth * height * width)
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=-1)
        value = self.value_conv(x).view(batch_size, -1, depth * height * width)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, depth, height, width)
        return self.gamma * out + x


class ResidualProductFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.modal1_conv = nn.Conv3d(in_channels, in_channels, 1)
        self.modal2_conv = nn.Conv3d(in_channels, in_channels, 1)

    def forward(self, modal1, modal2):
        return self.modal1_conv(modal1) * self.modal2_conv(modal2)


class ImprovedFusionUnit(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super().__init__()
        self.fused = ResidualProductFusion(in_channels)
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, 3, padding=1),  # 输出通道数改为out_channels
            nn.InstanceNorm3d(in_channels),  # 通道数正确匹配
            nn.LeakyReLU(0.2),
            DoubleConv(in_channels, in_channels)
        )

    def forward(self, modal1, modal2):
        # Perform residual product fusion
        cross_features = self.fused(modal1, modal2)
        return self.conv(torch.cat([modal1, modal2], dim=1)) + cross_features


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, reduction_ratio=4):
        super().__init__()

        # # # 1. 增强的残差路径 (包含非线性变换)
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1),
            nn.InstanceNorm3d(out_channels),
            nn.GELU()
        ) if in_channels != out_channels else nn.Identity()

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels // 2, 3, padding=1),
            nn.InstanceNorm3d(out_channels // 2),
            nn.LeakyReLU(0.2),
            nn.Conv3d(out_channels // 2, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2)
        )
        self.se = SqueezeExcitation(out_channels, dimensionality=3)

    def forward(self, x):
        residual = self.shortcut(x)
        return self.se(self.conv(x) + residual)



class TreeEncoderLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, modal_numbers=3, dimension=3):
        super().__init__()
        self.modal_number = modal_numbers
        for index in range(self.modal_number):
            self.add_module(f"DoubleConv_{index}", DoubleConv(in_channels, out_channels))
        for index in range(self.modal_number - 1):
            self.add_module(f"FusionUnit_{index}", ImprovedFusionUnit(out_channels))

    def forward(self, x_list):
        results = []
        if len(x_list) >= 2:
            for index in range(self.modal_number - 1):
                modal1 = self._modules[f"DoubleConv_{index}"](x_list[index])
                modal2 = self._modules[f"DoubleConv_{index + 1}"](x_list[index + 1])
                results.append(self._modules[f"FusionUnit_{index}"](modal1, modal2))
        # else:
        #     results = [self._modules["DoubleConv_0"](x_list[0])]
        return results


class TreePoolLayer(nn.Module):
    def __init__(self, modal_numbers=3):
        super().__init__()
        self.modal_numbers = modal_numbers
        for index in range(modal_numbers):
            self.add_module(f"TreePoolLayer_{index}", nn.MaxPool3d(2, stride=2))

    def forward(self, x_list):
        return [self._modules[f"TreePoolLayer_{idx}"](x) for idx, x in enumerate(x_list)]


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, modal_numbers=4):
        super().__init__()
        self.out_channels = out_channels
        self.modal_numbers = modal_numbers


        self.channels = [8, 16, 32, 64]

        if modal_numbers == 4:
            # 编码器
            self.treelayer1 = TreeEncoderLayer(in_channels, self.channels[0], 4)
            self.pool1 = TreePoolLayer(3)

            self.treelayer2 = TreeEncoderLayer(self.channels[0], self.channels[1], 3)
            self.pool2 = TreePoolLayer(2)

            self.treelayer3 = TreeEncoderLayer(self.channels[1], self.channels[2], 2)
            self.pool3 = TreePoolLayer(1)

            self.bottleneck = nn.Sequential(
                # DoubleConv(self.channels[2], self.channels[3]),
                ASPP(self.channels[2], self.channels[3], rates=(1, 2, 4, 8)),
                # CrissCrossAttention(self.channels[3]),
                # TransformerBlock(self.channels[3], num_heads=8)
            )

            self.upconv3 = UpSampleBlock(self.channels[3], self.channels[2])
            self.upconv2 = UpSampleBlock(self.channels[2], self.channels[1])
            self.upconv1 = UpSampleBlock(self.channels[1], self.channels[0])

            self.fusion3 = SkipFusionBlock(self.channels[2], num_modals=1)
            self.fusion2 = SkipFusionBlock(self.channels[1], num_modals=2)
            self.fusion1 = SkipFusionBlock(self.channels[0], num_modals=3)

            # 输出层
            self.out_conv = nn.Sequential(
                nn.Conv3d(self.channels[0], out_channels, 1),
                nn.Sigmoid() if out_channels == 1 else nn.Softmax(dim=1)
            )

    def tensor_list_mean(self, tensor_list):
        if len(tensor_list)>1:
            return torch.sum(torch.stack(tensor_list,dim=0), dim=0)
        else:
            return tensor_list[0]

    def forward(self, x):
        x_lists = x.chunk(self.modal_numbers, 1)

        if self.modal_numbers == 4:

            tenc1 = self.treelayer1(x_lists)
            tenc2 = self.treelayer2(self.pool1(tenc1))
            tenc3 = self.treelayer3(self.pool2(tenc2))
            # tenc4 = self.treelayer4(self.pool3(tenc3))

            bottleneck = self.bottleneck(self.pool3(tenc3)[0])

            # dec4 = self.upconv4(bottleneck, self.tensor_list_mean(tenc4))
            # dec3 = self.upconv3(bottleneck, self.tensor_list_mean(tenc3))
            # dec2 = self.upconv2(dec3, self.tensor_list_mean(tenc2))
            # dec1 = self.upconv1(dec2, self.tensor_list_mean(tenc1))

            dec3 = self.upconv3(bottleneck, self.fusion3(tenc3))
            dec2 = self.upconv2(dec3, self.fusion2(tenc2))
            dec1 = self.upconv1(dec2, self.fusion1(tenc1))

            return self.out_conv(dec1)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet3D(out_channels=4, modal_numbers=4).to(device)

    # 测试显存占用
    input_data = torch.randn(1, 4, 128, 128, 128).to(device)
    print("memory:", torch.cuda.memory_allocated() / 1024 ** 3, "GB")

    output = model(input_data)
    print("forward memory:", torch.cuda.memory_allocated() / 1024 ** 3, "GB")
    print("output shape:", output.shape)
