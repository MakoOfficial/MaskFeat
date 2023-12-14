import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class HOGLayerC(nn.Module):
    def __init__(self, nbins=9, pool=8, cell_per_blocks=2):
        super(HOGLayerC, self).__init__()
        self.nbins = nbins
        self.pool = pool
        self.cell_per_blocks = cell_per_blocks
        self.pi = math.pi
        weight_x = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        weight_x = weight_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        weight_y = weight_x.transpose(2, 3)
        self.weight_x = weight_x
        self.weight_y = weight_y

    @torch.no_grad()
    def forward(self, x):
        # input is RGB image with shape [B 3 H W]
        x = F.pad(x, pad=(1, 1, 1, 1), mode="reflect")
        gx_rgb = F.conv2d(
            x, self.weight_x, bias=None, stride=1, padding=0, groups=3
        )
        gy_rgb = F.conv2d(
            x, self.weight_y, bias=None, stride=1, padding=0, groups=3
        )
        norm_rgb = torch.stack([gx_rgb, gy_rgb], dim=-1).norm(dim=-1)
        phase = torch.atan2(gx_rgb, gy_rgb)
        phase = phase / self.pi * self.nbins  # [-9, 9]

        b, c, h, w = norm_rgb.shape
        out = torch.zeros(
            (b, c, self.nbins, h, w), dtype=torch.float, device=x.device
        )
        phase = phase.view(b, c, 1, h, w)
        norm_rgb = norm_rgb.view(b, c, 1, h, w)

        out.scatter_add_(2, phase.floor().long() % self.nbins, norm_rgb)

        out = out.unfold(3, self.pool, self.pool)
        out = out.unfold(4, self.pool, self.pool)
        out = out.sum(dim=[-1, -2])

        out = F.normalize(out, p=2, dim=2)

        hog = out.flatten(1, 2)
        hog = (
            hog.permute(0, 2, 3, 1)
            .unfold(1, self.cell_per_blocks, self.cell_per_blocks)
            .unfold(2, self.cell_per_blocks, self.cell_per_blocks)
            .flatten(1, 2)
            .flatten(2)
        )


        return hog  # B N nbins*channel*cell_per_blocks**2
