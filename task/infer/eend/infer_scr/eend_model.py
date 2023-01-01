"""EEND model module."""
from contextlib import contextmanager
from distutils.version import LooseVersion
from itertools import permutations
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from typeguard import check_argument_types

from ntu_eend.scr.nnet.backend.nets_utils import to_device
from ntu_eend.scr.nnet.asr.encoder.abs_encoder import AbsEncoder
from ntu_eend.scr.nnet.asr.frontend.abs_frontend import AbsFrontend
from ntu_eend.scr.nnet.asr.specaug.abs_specaug import AbsSpecAug
from ntu_eend.scr.nnet.diar.attractor.abs_attractor import AbsAttractor
from ntu_eend.scr.nnet.enh.decoder.abs_decoder import AbsDecoder
from ntu_eend.scr.task.layers.abs_normalize import AbsNormalize
from ntu_eend.scr.utils.device_funcs import force_gatherable
from ntu_eend.scr.task.train.abs_espnet_model import AbsESPnetModel




class EENDModel(AbsESPnetModel):
    """Speaker Diarization model

    If "attractor" is "None", SA-EEND will be used.
    Else if "attractor" is not "None", EEND-EDA will be used.
    For the details about SA-EEND and EEND-EDA, refer to the following papers:
    SA-EEND: https://arxiv.org/pdf/1909.06247.pdf
    EEND-EDA: https://arxiv.org/pdf/2005.09921.pdf, https://arxiv.org/pdf/2106.10654.pdf
    """

    def __init__(
        self,
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        label_aggregator: torch.nn.Module,
        encoder: AbsEncoder,
        decoder: AbsDecoder,
        attractor: Optional[AbsAttractor],
        attractor_weight: float = 1.0,
    ):
        assert check_argument_types()

        super().__init__()

        self.encoder = encoder
        self.normalize = normalize
        self.frontend = frontend
        self.specaug = specaug
        self.label_aggregator = label_aggregator
        self.attractor_weight = attractor_weight
        self.attractor = attractor
        self.decoder = decoder

        if self.attractor is not None:
            self.decoder = None
        elif self.decoder is not None:
            self.num_spk = decoder.num_spk
        else:
            raise NotImplementedError

    def forward(
        self,
        speech: torch.Tensor,
    ) -> torch.Tensor:

        batch_size = speech.shape[0]
        speech_lengths = torch.ones(batch_size).int().fill_(speech.shape[1])

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        
            # 2a. Decoder (baiscally a predction layer after encoder_out)
        pred = self.decoder(encoder_out, encoder_out_lens)
        
        return pred

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        spk_labels: torch.Tensor = None,
        spk_labels_lengths: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        pass

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch,)
        """
        
        # 1. Extract feats
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)

        # 2. Data augmentation
        if self.specaug is not None and self.training:
            feats, feats_lengths = self.specaug(feats, feats_lengths)

        # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
        if self.normalize is not None:
            feats, feats_lengths = self.normalize(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim)
        encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        return encoder_out, encoder_out_lens

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = speech.shape[0]
        speech_lengths = (
            speech_lengths
            if speech_lengths is not None
            else torch.ones(batch_size).int() * speech.shape[1]
        )

        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths




    