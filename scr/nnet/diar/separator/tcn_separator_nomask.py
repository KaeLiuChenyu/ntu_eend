from distutils.version import LooseVersion
from typing import Tuple
from typing import Union

import torch
from torch_complex.tensor import ComplexTensor

from ntu_eend.scr.nnet.diar.layers.tcn_nomask import TemporalConvNet
from ntu_eend.scr.nnet.diar.separator.abs_separator import AbsSeparator
from ntu_eend.scr.nnet.enh.layers.complex_utils import is_complex

is_torch_1_9_plus = LooseVersion(torch.__version__) >= LooseVersion("1.9.0")


class TCNSeparator(AbsSeparator):
    def __init__(
        self,
        input_dim: int = 512,
        layer: int = 8,
        stack: int = 3,
        bottleneck_dim: int = 128,
        hidden_dim: int = 512,
        kernel: int = 3,
        causal: bool = False,
        norm_type: str = "gLN",
    ):
        """Temporal Convolution Separator

        Args:
            input_dim: input feature dimension
            layer: int, number of layers in each stack.
            stack: int, number of stacks
            bottleneck_dim: bottleneck dimension
            hidden_dim: number of convolution channel
            kernel: int, kernel size.
            causal: bool, defalut False.
            norm_type: str, choose from 'BN', 'gLN', 'cLN'
        """
        super().__init__()

        self.tcn = TemporalConvNet(
            N=input_dim,
            B=bottleneck_dim,
            H=hidden_dim,
            P=kernel,
            X=layer,
            R=stack,
            norm_type=norm_type,
            causal=causal,
        )

        self._output_dim = bottleneck_dim

    def forward(
        self, input: Union[torch.Tensor, ComplexTensor], ilens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward.

        Args:
            input (torch.Tensor or ComplexTensor): Encoded feature [B, T, N]
            ilens (torch.Tensor): input lengths [Batch]

        Returns:
            feats (torch.Tensor): [B, T, bottleneck_dim]
            ilens (torch.Tensor): (B,)
        """
        # if complex spectrum
        if is_complex(input):
            feature = abs(input)
        else:
            feature = input
        feature = feature.transpose(1, 2)  # B, N, L

        feats = self.tcn(feature)  # [B, bottleneck_dim, L]
        feats = feats.transpose(1, 2)  # B, L, bottleneck_dim

        return feats, ilens

    @property
    def output_dim(self) -> int:
        return self._output_dim
