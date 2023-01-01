"""EEND_SS model module."""
from contextlib import contextmanager
from distutils.version import LooseVersion
from itertools import permutations
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from typeguard import check_argument_types


from ntu_eend.scr.nnet.backend.nets_utils import to_device
from ntu_eend.scr.nnet.asr.encoder.abs_encoder import AbsEncoder as AbsDiarEncoder

from ntu_eend.scr.nnet.diar.attractor.abs_attractor import AbsAttractor
from ntu_eend.scr.nnet.diar.decoder.abs_decoder import AbsDecoder as AbsDiarDecoder
from ntu_eend.scr.nnet.diar.layers.abs_mask import AbsMask
from ntu_eend.scr.nnet.diar.separator.abs_separator import AbsSeparator
from ntu_eend.scr.nnet.enh.decoder.abs_decoder import AbsDecoder as AbsEnhDecoder
from ntu_eend.scr.nnet.enh.encoder.abs_encoder import AbsEncoder as AbsEnhEncoder
from ntu_eend.scr.nnet.enh.loss.criterions.tf_domain import FrequencyDomainLoss
from ntu_eend.scr.nnet.enh.loss.criterions.time_domain import TimeDomainLoss
from ntu_eend.scr.nnet.enh.loss.wrappers.abs_wrapper import AbsLossWrapper
from ntu_eend.scr.task.layers.abs_normalize import AbsNormalize
from ntu_eend.scr.utils.device_funcs import force_gatherable
from ntu_eend.scr.task.train.abs_espnet_model import AbsESPnetModel
from ntu_eend.scr.nnet.asr.frontend.abs_frontend import AbsFrontend
from ntu_eend.scr.nnet.asr.specaug.abs_specaug import AbsSpecAug


EPS = torch.finfo(torch.get_default_dtype()).eps


class EENDSSModel(AbsESPnetModel):
    """Joint Speech Diarization & Separation model"""

    def __init__(
        self,
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        label_aggregator: torch.nn.Module,
        diar_encoder: AbsDiarEncoder,
        diar_decoder: AbsDiarDecoder,
        attractor: Optional[AbsAttractor],
        enh_encoder: AbsEnhEncoder,
        separator: AbsSeparator,
        mask_module: AbsMask,
        enh_decoder: AbsEnhDecoder,
        loss_wrappers: List[AbsLossWrapper],
        stft_consistency: bool = False,
        loss_type: str = "mask_mse",
        mask_type: Optional[str] = None,
        enh_weight: float = 1.0,
        diar_weight: float = 1.0,
        attractor_weight: float = 1.0,
        pooling_kernel: int = 1,
        num_spks : int = 2
    ):
        assert check_argument_types()

        super().__init__()

        # Main modules
        self.diar_encoder = diar_encoder
        self.attractor = attractor
        self.enh_encoder = enh_encoder
        self.separator = separator
        self.mask_module = mask_module
        self.enh_decoder = enh_decoder

        # Additional information
        self.num_spks = num_spks

    def forward(
        self,
        speech_mix: torch.Tensor,
    ) -> torch.Tensor:
		# Get speech lengths
        batch_size = speech_mix.shape[0]
        speech_lengths = torch.ones(batch_size).int().fill_(speech_mix.shape[1])
        
        # Enh model forward
        feature_mix, flens = self.enh_encoder(_input = speech_mix, 
                  ilens = speech_lengths)

        bottleneck_feats, flens = self.separator(_input = feature_mix, 
                  ilens = flens)

        feature_pre, flens, _ = self.mask_module(_input = feature_mix, 
                  ilens = flens, 
                  bottleneck_feat = bottleneck_feats, 
                  num_spk = self.num_spks)


        # Diar model forward	
        encoder_out, encoder_out_lens, _ = self.diar_encoder(xs_pad = bottleneck_feats, 
                      ilens = flens)


        # Shuffle samples
        encoder_out_shuffled = encoder_out.clone()
        for i, out_len in enumerate(encoder_out_lens):
          shuff_out_len = torch.randperm(out_len)
          encoder_out_shuffled[i, :out_len, :] = encoder_out[i, shuff_out_len, :]


        # Attractor forward
        input_tensor = torch.zeros(encoder_out.size(0),
              self.num_spks + 1, 
              encoder_out.size(2))

        decoder_input = to_device(m = self, x = input_tensor)

        attractor, _ = self.attractor(enc_input = encoder_out_shuffled,
                ilens = encoder_out_lens,
                dec_input = decoder_input)


        # Remove the final attractor which does not correspond to a speaker
        # Then multiply the attractors and encoder_out
        diar_pred = torch.bmm(encoder_out, attractor[:, :-1, :].permute(0, 2, 1))

        # Get prediction from enhancement decoder
        if feature_pre is not None:
          speech_pred = [self.enh_decoder(ps, speech_lengths)[0] for ps in feature_pre]
        else:
          speech_pred = None

        return diar_pred, speech_pred


    def collect_feats(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: torch.Tensor,
        spk_labels: torch.Tensor = None,
        spk_labels_lengths: torch.Tensor = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        pass
