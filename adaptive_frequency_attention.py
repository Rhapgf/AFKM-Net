import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math
class LearnableFrequencyFilters(nn.Module):
    def __init__(
            self,
            channels: int,
            num_filters: int = 4,
            kernel_sizes: Tuple[int, ...] = (1, 3, 5, 7)
    ):
        super().__init__()

        self.channels = channels
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes

        assert len(kernel_sizes) == num_filters, \
            f"Number of kernel sizes must match num_filters"
        self.freq_filters = nn.ModuleList()
        for kernel_size in kernel_sizes:
            padding = kernel_size // 2
            filter_module = nn.Sequential(
                nn.Conv2d(
                    channels, channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    groups=channels,
                    bias=False
                ),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels)
            )
            self.freq_filters.append(filter_module)
        self._initialize_frequency_response()

    def _initialize_frequency_response(self):
        for idx, (kernel_size, filter_module) in enumerate(
                zip(self.kernel_sizes, self.freq_filters)
        ):
            depthwise_conv = filter_module[0]

            with torch.no_grad():
                if kernel_size >= 5:
                    for i in range(self.channels):
                        sigma = kernel_size / 4.0
                        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
                        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
                        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
                        kernel = kernel / kernel.sum()
                        depthwise_conv.weight[i, 0] = kernel

                elif kernel_size == 1:
                    nn.init.ones_(depthwise_conv.weight)

                else:
                    nn.init.kaiming_normal_(depthwise_conv.weight, mode='fan_out')

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        freq_features = []
        for filter_module in self.freq_filters:
            freq_feat = filter_module(x)
            freq_features.append(freq_feat)

        return tuple(freq_features)


class AdaptiveFrequencySelector(nn.Module):
    def __init__(
            self,
            channels: int,
            num_filters: int = 4,
            reduction: int = 16
    ):
        super().__init__()

        self.channels = channels
        self.num_filters = num_filters

        reduced_dim = max(channels // reduction, 16)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.gate_network = nn.Sequential(
            nn.Conv2d(channels * num_filters, reduced_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_dim, num_filters, 1),
            nn.Softmax(dim=1)
        )
        self.channel_attention = nn.Sequential(
            nn.Conv2d(channels * num_filters, reduced_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_dim, channels, 1),
            nn.Sigmoid()
        )
    def forward(
            self,
            freq_features: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = freq_features[0].shape[0]
        freq_concat = torch.cat(freq_features, dim=1)
        global_feat = self.global_pool(freq_concat)
        freq_weights = self.gate_network(global_feat)
        channel_att = self.channel_attention(global_feat)
        return freq_weights, channel_att
class FrequencyGuidedSpatialAttention(nn.Module):
    def __init__(
            self,
            num_filters: int = 4,
            kernel_size: int = 7
    ):
        super().__init__()

        self.num_filters = num_filters
        padding = kernel_size // 2
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(num_filters * 2, num_filters, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, 1, kernel_size, padding=padding),
            nn.Sigmoid()
        )

    def forward(
            self,
            freq_features: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        spatial_stats = []

        for freq_feat in freq_features:
            avg_out = torch.mean(freq_feat, dim=1, keepdim=True)
            max_out, _ = torch.max(freq_feat, dim=1, keepdim=True)
            spatial_stats.extend([avg_out, max_out])

        spatial_concat = torch.cat(spatial_stats, dim=1)

        spatial_att = self.spatial_conv(spatial_concat)

        return spatial_att


class AFSA(nn.Module):
    def __init__(
            self,
            channels: int,
            num_filters: int = 4,
            kernel_sizes: Tuple[int, ...] = (1, 3, 5, 7),
            reduction: int = 16,
            spatial_kernel: int = 7
    ):
        super().__init__()

        self.channels = channels
        self.num_filters = num_filters

        self.freq_filters = LearnableFrequencyFilters(
            channels, num_filters, kernel_sizes
        )

        self.freq_selector = AdaptiveFrequencySelector(
            channels, num_filters, reduction
        )

        self.spatial_attention = FrequencyGuidedSpatialAttention(
            num_filters, spatial_kernel
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(channels * num_filters, channels, 1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        freq_features = self.freq_filters(x)
        freq_weights, channel_att = self.freq_selector(freq_features)
        weighted_features = []
        for i, freq_feat in enumerate(freq_features):
            weight = freq_weights[:, i:i + 1, :, :]
            weighted_feat = freq_feat * weight
            weighted_features.append(weighted_feat)
        freq_concat = torch.cat(weighted_features, dim=1)
        freq_fused = self.fusion(freq_concat)
        x_channel = freq_fused * channel_att
        spatial_att = self.spatial_attention(freq_features)
        x_spatial = x_channel * spatial_att
        out = x_spatial + x
        return out

if __name__ == "__main__":
    test_afsa()