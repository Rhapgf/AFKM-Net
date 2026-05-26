import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from model.vmamba.vmamba import VSSBlock, LayerNorm2d
class HermiteKANLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        degree: int = 5,
        base_activation: nn.Module = nn.SiLU(),
        use_layernorm: bool = True
    ):
        super(HermiteKANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.input_norm = nn.LayerNorm(in_features)
        self.base_linear = nn.Linear(in_features, out_features)
        self.base_activation = base_activation
        self.hermite_coeffs = nn.Parameter(
            torch.zeros(out_features, in_features, degree + 1)
        )
        self.hermite_scale = nn.Parameter(torch.tensor(0.1))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_linear.weight, a=math.sqrt(5))
        nn.init.zeros_(self.base_linear.bias)
        with torch.no_grad():
            for i in range(self.degree + 1):
                scale = 0.1 / (1 + i)
                nn.init.normal_(self.hermite_coeffs[:, :, i], mean=0.0, std=scale)

    def hermite_polynomials(self, x: torch.Tensor) -> torch.Tensor:
        He0 = torch.ones_like(x)
        He1 = x

        hermite_polys = [He0, He1]
        for n in range(2, self.degree + 1):
            He_n = x * hermite_polys[-1] - (n - 1) * hermite_polys[-2]
            hermite_polys.append(He_n)
        return torch.stack(hermite_polys, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)
        base_output = self.base_linear(self.base_activation(x_flat))
        if self.use_layernorm:
            x_norm = self.input_norm(x_flat)
        else:
            x_norm = (x_flat - x_flat.mean(dim=-1, keepdim=True)) / (x_flat.std(dim=-1, keepdim=True) + 1e-6)
        hermite_basis = self.hermite_polynomials(x_norm)
        hermite_output = torch.einsum('oid,bid->bo', self.hermite_coeffs, hermite_basis)
        output = base_output + self.hermite_scale * hermite_output

        return output.reshape(*batch_shape, self.out_features)


class HermiteKANPath(nn.Module):
    def __init__(
        self,
        dim: int,
        degree: int = 7,
        reduction_ratio: int = 4
    ):
        super(HermiteKANPath, self).__init__()
        self.dim = dim
        self.degree = degree
        hidden_dim = max(dim // reduction_ratio, 16)
        self.hidden_dim = hidden_dim
        self.reduction = nn.Linear(dim, hidden_dim)
        self.kan_layer = nn.Sequential(
            HermiteKANLayer(
                hidden_dim,
                hidden_dim,
                degree=degree,
                use_layernorm=True
            ),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim)
        )
        self.output_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x_reduced = self.reduction(x)
        x_kan = self.kan_layer(x_reduced)
        output = self.output_proj(x_kan)
        output = identity + self.output_scale * output
        return output
class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, dim: int, num_branches: int = 2):
        super(AdaptiveFeatureFusion, self).__init__()
        self.dim = dim
        self.num_branches = num_branches

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.weight_net = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, num_branches),
            nn.Softmax(dim=-1)
        )

        self.refine = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1)
        )

    def forward(self, features: list) -> torch.Tensor:
        assert len(features) == self.num_branches

        B, H, W, C = features[0].shape

        features_bchw = [f.permute(0, 3, 1, 2) for f in features]

        global_context = self.global_pool(features_bchw[0]).squeeze(-1).squeeze(-1)

        branch_weights = self.weight_net(global_context)

        fused = sum(
            w.view(B, 1, 1, 1) * f
            for w, f in zip(branch_weights.unbind(-1), features_bchw)
        )

        refined = self.refine(fused)

        output = refined.permute(0, 2, 3, 1)

        return output

class HK-VSS(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        drop_path: float = 0.0,
        kan_degree: int = 7,
        kan_reduction: int = 4,
        use_vss: bool = True
    ):
        super(HK-VSS, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_vss = use_vss and VMAMBA_AVAILABLE
        if VMAMBA_AVAILABLE:
            self.norm = LayerNorm2d(hidden_dim)
        else:
            self.norm = nn.BatchNorm2d(hidden_dim)
        if self.use_vss:
            self.vss_branch = VSSBlock(
                hidden_dim=hidden_dim,
                drop_path=drop_path
            )
        else:
            self.vss_branch = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, 1)
            )
        self.kan_branch = HermiteKANPath(
            dim=hidden_dim,
            degree=kan_degree,
            reduction_ratio=kan_reduction
        )
        self.fusion = AdaptiveFeatureFusion(
            dim=hidden_dim,
            num_branches=2
        )
        self.drop_path = nn.Identity()
        if drop_path > 0:
            try:
                from timm.models.layers import DropPath
                self.drop_path = DropPath(drop_path)
            except:
                pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x_bchw = x.permute(0, 3, 1, 2)
        x_norm = self.norm(x_bchw)
        x_bhwc = x_norm.permute(0, 2, 3, 1)
        if self.use_vss:
            branch1_out = self.vss_branch(x_bhwc)
        else:
            branch1_out = self.vss_branch(x_norm).permute(0, 2, 3, 1)
        branch2_out = self.kan_branch(x_bhwc)
        fused = self.fusion([branch1_out, branch2_out])
        output = shortcut + self.drop_path(fused)
        return output
