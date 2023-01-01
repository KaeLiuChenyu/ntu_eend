import argparse
from typing import Callable
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from typeguard import check_argument_types
from typeguard import check_return_type


# Diar Encoder
from ntu_eend.scr.nnet.asr.encoder.abs_encoder import AbsEncoder as DiarAbsEncoder
from ntu_eend.scr.nnet.asr.encoder.transformer_encoder import TransformerEncoder

# Attractor
from ntu_eend.scr.nnet.diar.attractor.abs_attractor import AbsAttractor
from ntu_eend.scr.nnet.diar.attractor.rnn_attractor import RnnAttractor

# Frontend
from ntu_eend.scr.nnet.asr.frontend.abs_frontend import AbsFrontend
from ntu_eend.scr.nnet.asr.frontend.default import DefaultFrontend

# SpecAug
from ntu_eend.scr.nnet.asr.specaug.abs_specaug import AbsSpecAug
from ntu_eend.scr.nnet.asr.specaug.specaug import SpecAug

# Normalize
from ntu_eend.scr.task.layers.abs_normalize import AbsNormalize
from ntu_eend.scr.task.layers.global_mvn import GlobalMVN
from ntu_eend.scr.task.layers.utterance_mvn import UtteranceMVN

# Diar Decoder
from ntu_eend.scr.nnet.diar.decoder.abs_decoder import AbsDecoder as DiarAbsDecoder
from ntu_eend.scr.nnet.diar.decoder.linear_decoder import LinearDecoder

# Model
from ntu_eend.scr.nnet.diar.eend_ss_model import EENDSSModel

# Mask
from ntu_eend.scr.nnet.diar.layers.abs_mask import AbsMask
from ntu_eend.scr.nnet.diar.layers.multi_mask import MultiMask

# Diar Loss
from ntu_eend.scr.nnet.diar.loss.wrappers.pit_solver_2 import PITSolver2

# Separator
from ntu_eend.scr.nnet.diar.separator.abs_separator import AbsSeparator
from ntu_eend.scr.nnet.diar.separator.tcn_separator_nomask import TCNSeparator

# Enh Decoder
from ntu_eend.scr.nnet.enh.decoder.abs_decoder import AbsDecoder as EnhAbsDecoder
from ntu_eend.scr.nnet.enh.decoder.conv_decoder import ConvDecoder

# Enh Encoder
from ntu_eend.scr.nnet.enh.encoder.abs_encoder import AbsEncoder as EnhAbsEncoder
from ntu_eend.scr.nnet.enh.encoder.conv_encoder import ConvEncoder

# Enh Loss
from ntu_eend.scr.nnet.enh.loss.criterions.abs_loss import AbsEnhLoss
from ntu_eend.scr.nnet.enh.loss.criterions.time_domain import SISNRLoss

# Enh loss wrapper
from ntu_eend.scr.nnet.enh.loss.wrappers.abs_wrapper import AbsLossWrapper
from ntu_eend.scr.nnet.enh.loss.wrappers.pit_solver import PITSolver


from ntu_eend.scr.task.layers.label_aggregation import LabelAggregate
from ntu_eend.scr.task.abs_task import AbsTask
from ntu_eend.scr.task.train.class_choices import ClassChoices
from ntu_eend.scr.task.train.collate_fn import CommonCollateFn
from ntu_eend.scr.task.train.preprocessor import CommonPreprocessor
from ntu_eend.scr.task.train.trainer import Trainer


from ntu_eend.scr.utils.initialize import initialize
from ntu_eend.scr.utils.get_default_kwargs import get_default_kwargs
from ntu_eend.scr.utils.nested_dict_action import NestedDictAction
from ntu_eend.scr.utils.types import int_or_none
from ntu_eend.scr.utils.types import str2bool
from ntu_eend.scr.utils.types import str_or_none


frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(
        default=DefaultFrontend,
    ),
    type_check=AbsFrontend,
    default=None,
    optional=True,
)
specaug_choices = ClassChoices(
    name="specaug",
    classes=dict(specaug=SpecAug),
    type_check=AbsSpecAug,
    default=None,
    optional=True,
)

normalize_choices = ClassChoices(
    "normalize",
    classes=dict(
        global_mvn=GlobalMVN,
        utterance_mvn=UtteranceMVN,
    ),
    type_check=AbsNormalize,
    default=None,
    optional=True,
)

label_aggregator_choices = ClassChoices(
    "label_aggregator",
    classes=dict(label_aggregator=LabelAggregate),
    default="label_aggregator",
)

diar_encoder_choices = ClassChoices(
    "diar_encoder",
    classes=dict(
        transformer=TransformerEncoder,
    ),
    type_check=DiarAbsEncoder,
    default="transformer",
)

diar_decoder_choices = ClassChoices(
    "diar_decoder",
    classes=dict(linear=LinearDecoder),
    type_check=DiarAbsDecoder,
    default="linear",
)

attractor_choices = ClassChoices(
    "attractor",
    classes=dict(
        rnn=RnnAttractor,
    ),
    type_check=AbsAttractor,
    default=None,
    optional=True,
)

enh_encoder_choices = ClassChoices(
    name="enh_encoder",
    classes=dict(conv=ConvEncoder),
    type_check=EnhAbsEncoder,
    default="stft",
)

separator_choices = ClassChoices(
    name="separator",
    classes=dict(
        tcn=TCNSeparator,
    ),
    type_check=AbsSeparator,
    default="tcn",
)

mask_module_choices = ClassChoices(
    name="mask_module",
    classes=dict(multi_mask=MultiMask),
    type_check=AbsMask,
    default="multi_mask",
)

enh_decoder_choices = ClassChoices(
    name="enh_decoder",
    classes=dict(conv=ConvDecoder,),
    type_check=EnhAbsDecoder,
    default="stft",
)

loss_wrapper_choices = ClassChoices(
    name="loss_wrappers",
    classes=dict(pit=PITSolver, pit2=PITSolver2),
    type_check=AbsLossWrapper,
    default=None,
)

criterion_choices = ClassChoices(
    name="criterions",
    classes=dict(
        si_snr=SISNRLoss,
    ),
    type_check=AbsEnhLoss,
    default=None,
)

MAX_REFERENCE_NUM = 100


class EENDSSTask(AbsTask):
    # If you need more than one optimizer, change this value
    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [
        # --frontend and --frontend_conf
        frontend_choices,
        # --specaug and --specaug_conf
        specaug_choices,
        # --normalize and --normalize_conf
        normalize_choices,
        # --diar_encoder and --diar_encoder_conf
        diar_encoder_choices,
        # --diar_decoder and --diar_decoder_conf
        diar_decoder_choices,
        # --label_aggregator and --label_aggregator_conf
        label_aggregator_choices,
        # --attractor and --attractor_conf
        attractor_choices,
        # --enh_encoder and --enh_encoder_conf
        enh_encoder_choices,
        # --separator and --separator_conf
        separator_choices,
        # --mask_module and --mask_module_conf
        mask_module_choices,
        # --enh_decoder and --enh_decoder_conf
        enh_decoder_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")

        group.add_argument(
            "--num_spk",
            type=int_or_none,
            default=None,
            help="The number fo speakers (for each recording) used in system training",
        )

        group.add_argument(
            "--init",
            type=lambda x: str_or_none(x.lower()),
            default=None,
            help="The initialization method",
            choices=[
                "chainer",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
                None,
            ],
        )

        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(EENDSSModel),
            help="The keyword arguments for model class.",
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=True,
            help="Apply preprocessing to data or not",
        )

        group.add_argument(
            "--criterions",
            action=NestedDictAction,
            default=[
                {
                    "name": "si_snr",
                    "conf": {},
                    "wrapper": "fixed_order",
                    "wrapper_conf": {},
                },
            ],
            help="The criterions binded with the loss wrappers.",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    def build_collate_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        assert check_argument_types()
        # NOTE(kamo): int value = 0 is reserved by CTC-blank symbol
        return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)

    @classmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()
        if args.use_preprocessor:
            # FIXME (jiatong): add more argument here
            retval = CommonPreprocessor(train=train)
        else:
            retval = None
        assert check_return_type(retval)
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("speech_mix", "spk_labels", "speech_ref1")
        else:
            # Recognition mode
            retval = ("speech_mix",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ["dereverb_ref{}".format(n) for n in range(1, MAX_REFERENCE_NUM + 1)]
        retval += ["speech_ref{}".format(n) for n in range(2, MAX_REFERENCE_NUM + 1)]
        retval += ["noise_ref{}".format(n) for n in range(1, MAX_REFERENCE_NUM + 1)]
        retval = tuple(retval)
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> EENDSSModel:
        assert check_argument_types()

        # 1. Label Aggregator layer
        label_aggregator_class = label_aggregator_choices.get_class(
            args.label_aggregator
        )
        label_aggregator = label_aggregator_class(**args.label_aggregator_conf)

        # enh_encoder
        enh_encoder = enh_encoder_choices.get_class(args.enh_encoder)(
            **args.enh_encoder_conf
        )

        # separator
        separator = separator_choices.get_class(args.separator)(
            enh_encoder.output_dim, **args.separator_conf
        )
        input_size = separator.output_dim

        # mask_module
        mask_module_class = mask_module_choices.get_class(args.mask_module)
        mask_module = mask_module_class(
            input_dim=enh_encoder.output_dim,
            **args.mask_module_conf,
        )

        # enh_decoder
        enh_decoder = enh_decoder_choices.get_class(args.enh_decoder)(
            **args.enh_decoder_conf
        )

        # 2. frontend
        if args.frontend is not None:
            frontend_class = frontend_choices.get_class(args.frontend)
            frontend = frontend_class(**args.frontend_conf)
            input_size = input_size + frontend.output_size()
        else:
            frontend = None

        # 3. Data augmentation for spectrogram
        if args.specaug is not None:
            specaug_class = specaug_choices.get_class(args.specaug)
            specaug = specaug_class(**args.specaug_conf)
        else:
            specaug = None

        # 4. Normalization layer
        if args.normalize is not None:
            normalize_class = normalize_choices.get_class(args.normalize)
            normalize = normalize_class(**args.normalize_conf)
        else:
            normalize = None

        # 5. diar_encoder
        diar_encoder_class = diar_encoder_choices.get_class(args.diar_encoder)
        # Note(jiatong): Diarization may not use subsampling when processing
        diar_encoder = diar_encoder_class(
            input_size=input_size,
            **args.diar_encoder_conf,
        )

        # 6a. diar_decoder
        diar_decoder_class = diar_decoder_choices.get_class(args.diar_decoder)
        diar_decoder = diar_decoder_class(
            num_spk=args.num_spk,
            encoder_output_size=diar_encoder.output_size(),
            **args.diar_decoder_conf,
        )

        # 6b. Attractor
        if getattr(args, "attractor", None) is not None:
            attractor_class = attractor_choices.get_class(args.attractor)
            attractor = attractor_class(
                encoder_output_size=diar_encoder.output_size(),
                **args.attractor_conf,
            )
        else:
            attractor = None

        # loss_wrapper
        loss_wrappers = []
        for ctr in args.criterions:
            criterion = criterion_choices.get_class(ctr["name"])(**ctr["conf"])
            loss_wrapper = loss_wrapper_choices.get_class(ctr["wrapper"])(
                criterion=criterion, **ctr["wrapper_conf"]
            )
            loss_wrappers.append(loss_wrapper)

        # 7. Build model
        model = EENDSSModel(
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            label_aggregator=label_aggregator,
            diar_encoder=diar_encoder,
            diar_decoder=diar_decoder,
            attractor=attractor,
            enh_encoder=enh_encoder,
            separator=separator,
            mask_module=mask_module,
            enh_decoder=enh_decoder,
            loss_wrappers=loss_wrappers,
            **args.model_conf,
        )

        # FIXME(kamo): Should be done in model?
        # 8. Initialize
        if args.init is not None:
            initialize(model, args.init)

        assert check_return_type(model)
        return model


