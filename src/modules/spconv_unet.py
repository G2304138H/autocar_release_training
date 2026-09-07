"""spconv 2.x ports of the sparse 3-D U-Nets used by AutoCAR.

Both the released-code ``MinkUNet18A`` and paper-specified ``MinkUNet34C``
topologies are available using prebuilt ``spconv-cu124`` operators. This keeps
the primary CUDA 12.4 training environment independent of a source-built
MinkowskiEngine.
"""

from __future__ import annotations

import math

import torch
from torch import nn

try:
    import spconv.pytorch as spconv
except ImportError as exc:  # pragma: no cover - exercised in CUDA environment
    raise ImportError(
        "The SpconvUNet models require spconv 2.x. Install the wheel matching the "
        "PyTorch CUDA runtime (for example spconv-cu124)."
    ) from exc


def _replace_feature(tensor, features):
    """Use the spconv 2.x feature replacement API."""

    return tensor.replace_feature(features)


def _cat_same_coordinates(left, right):
    """Concatenate sparse tensors whose indice-key path guarantees alignment."""

    if left.indices.shape != right.indices.shape or not torch.equal(
        left.indices, right.indices
    ):
        raise RuntimeError(
            "Sparse skip tensors are not coordinate-aligned. This normally "
            "indicates inconsistent spconv indice keys or input coordinates."
        )
    return _replace_feature(left, torch.cat([left.features, right.features], dim=1))


class SparseBasicBlock(nn.Module):
    """Two submanifold convolutions with a residual projection when needed."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, indice_key: str):
        super().__init__()
        self.conv1 = spconv.SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            indice_key=indice_key,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.conv2 = spconv.SubMConv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            indice_key=indice_key,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.projection = (
            spconv.SubMConv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
                indice_key=f"{indice_key}_projection",
            )
            if in_channels != out_channels
            else None
        )
        self.projection_bn = (
            nn.BatchNorm1d(out_channels) if self.projection is not None else None
        )

    def forward(self, sparse_tensor):
        identity = sparse_tensor
        out = self.conv1(sparse_tensor)
        out = _replace_feature(out, self.activation(self.bn1(out.features)))
        out = self.conv2(out)
        out = _replace_feature(out, self.bn2(out.features))
        if self.projection is not None:
            identity = self.projection(identity)
            identity = _replace_feature(
                identity, self.projection_bn(identity.features)
            )
        out = _replace_feature(out, out.features + identity.features)
        return _replace_feature(out, self.activation(out.features))


class SparseResidualStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block_count: int,
        indice_key: str,
    ) -> None:
        super().__init__()
        blocks = [SparseBasicBlock(in_channels, out_channels, indice_key)]
        blocks.extend(
            SparseBasicBlock(out_channels, out_channels, indice_key)
            for _ in range(1, block_count)
        )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, sparse_tensor):
        for block in self.blocks:
            sparse_tensor = block(sparse_tensor)
        return sparse_tensor


class SpconvUNetBase(nn.Module):
    """Configurable spconv analogue of the released Minkowski U-Net."""

    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    INIT_DIM = 32

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        planes = self.PLANES
        layers = self.LAYERS
        self.activation = nn.ELU(inplace=True)

        self.conv0 = spconv.SubMConv3d(
            in_channels,
            self.INIT_DIM,
            kernel_size=5,
            padding=2,
            bias=False,
            # This 5x5 convolution and the 3x3 decoder convolutions operate on
            # the same coordinates, but they must not reuse one indice cache.
            # spconv requires a shared key to have identical kernel metadata
            # and algorithm selection.
            indice_key="input_subm5",
        )
        self.bn0 = nn.BatchNorm1d(self.INIT_DIM)

        self.down1 = spconv.SparseConv3d(
            self.INIT_DIM,
            self.INIT_DIM,
            kernel_size=2,
            stride=2,
            bias=False,
            indice_key="down1",
        )
        self.bn_down1 = nn.BatchNorm1d(self.INIT_DIM)
        self.stage1 = SparseResidualStage(
            self.INIT_DIM, planes[0], layers[0], "subm1"
        )

        self.down2 = spconv.SparseConv3d(
            planes[0],
            planes[0],
            kernel_size=2,
            stride=2,
            bias=False,
            indice_key="down2",
        )
        self.bn_down2 = nn.BatchNorm1d(planes[0])
        self.stage2 = SparseResidualStage(
            planes[0], planes[1], layers[1], "subm2"
        )

        self.down3 = spconv.SparseConv3d(
            planes[1],
            planes[1],
            kernel_size=2,
            stride=2,
            bias=False,
            indice_key="down3",
        )
        self.bn_down3 = nn.BatchNorm1d(planes[1])
        self.stage3 = SparseResidualStage(
            planes[1], planes[2], layers[2], "subm3"
        )

        self.down4 = spconv.SparseConv3d(
            planes[2],
            planes[2],
            kernel_size=2,
            stride=2,
            bias=False,
            indice_key="down4",
        )
        self.bn_down4 = nn.BatchNorm1d(planes[2])
        self.stage4 = SparseResidualStage(
            planes[2], planes[3], layers[3], "subm4"
        )

        self.up4 = spconv.SparseInverseConv3d(
            planes[3], planes[4], kernel_size=2, indice_key="down4", bias=False
        )
        self.bn_up4 = nn.BatchNorm1d(planes[4])
        self.stage5 = SparseResidualStage(
            planes[4] + planes[2], planes[4], layers[4], "subm3"
        )

        self.up5 = spconv.SparseInverseConv3d(
            planes[4], planes[5], kernel_size=2, indice_key="down3", bias=False
        )
        self.bn_up5 = nn.BatchNorm1d(planes[5])
        self.stage6 = SparseResidualStage(
            planes[5] + planes[1], planes[5], layers[5], "subm2"
        )

        self.up6 = spconv.SparseInverseConv3d(
            planes[5], planes[6], kernel_size=2, indice_key="down2", bias=False
        )
        self.bn_up6 = nn.BatchNorm1d(planes[6])
        self.stage7 = SparseResidualStage(
            planes[6] + planes[0], planes[6], layers[6], "subm1"
        )

        self.up7 = spconv.SparseInverseConv3d(
            planes[6], planes[7], kernel_size=2, indice_key="down1", bias=False
        )
        self.bn_up7 = nn.BatchNorm1d(planes[7])
        self.stage8 = SparseResidualStage(
            planes[7] + self.INIT_DIM, planes[7], layers[7], "subm0"
        )
        self.final = spconv.SubMConv3d(
            planes[7],
            out_channels,
            kernel_size=1,
            bias=True,
            indice_key="final",
        )
        self._initialise_weights()

    def _activate(self, sparse_tensor, batch_norm):
        return _replace_feature(
            sparse_tensor,
            self.activation(batch_norm(sparse_tensor.features)),
        )

    def _initialise_weights(self) -> None:
        sparse_convolution_types = (
            spconv.SubMConv3d,
            spconv.SparseConv3d,
            spconv.SparseInverseConv3d,
        )
        for module in self.modules():
            if isinstance(module, sparse_convolution_types):
                kernel_size = module.kernel_size
                if isinstance(kernel_size, int):
                    kernel_volume = kernel_size**3
                else:
                    kernel_volume = math.prod(int(value) for value in kernel_size)
                fan_out = int(module.out_channels) * kernel_volume
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=math.sqrt(2.0 / fan_out),
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, sparse_tensor):
        out = self._activate(self.conv0(sparse_tensor), self.bn0)
        skip0 = out

        out = self._activate(self.down1(out), self.bn_down1)
        skip1 = self.stage1(out)
        out = self._activate(self.down2(skip1), self.bn_down2)
        skip2 = self.stage2(out)
        out = self._activate(self.down3(skip2), self.bn_down3)
        skip3 = self.stage3(out)
        out = self._activate(self.down4(skip3), self.bn_down4)
        out = self.stage4(out)

        out = self._activate(self.up4(out), self.bn_up4)
        out = self.stage5(_cat_same_coordinates(out, skip3))
        out = self._activate(self.up5(out), self.bn_up5)
        out = self.stage6(_cat_same_coordinates(out, skip2))
        out = self._activate(self.up6(out), self.bn_up6)
        out = self.stage7(_cat_same_coordinates(out, skip1))
        out = self._activate(self.up7(out), self.bn_up7)
        out = self.stage8(_cat_same_coordinates(out, skip0))
        return self.final(out)


class SpconvUNet18A(SpconvUNetBase):
    """Port of the architecture hard-coded in the released Python model."""

    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)


class SpconvUNet34C(SpconvUNetBase):
    """Port of MinkUNet34C, the sparse backbone named in the paper."""

    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
