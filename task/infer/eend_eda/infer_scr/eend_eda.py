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

from ntu_eend.scr.nnet.asr.encoder.abs_encoder import AbsEncoder
from ntu_eend.scr.nnet.asr.encoder.transformer_encoder import TransformerEncoder
from ntu_eend.scr.nnet.diar.attractor.abs_attractor import AbsAttractor
from ntu_eend.scr.nnet.diar.attractor.rnn_attractor import RnnAttractor
from ntu_eend.scr.nnet.asr.frontend.abs_frontend import AbsFrontend
from ntu_eend.scr.nnet.asr.frontend.default import DefaultFrontend
from ntu_eend.scr.nnet.asr.specaug.abs_specaug import AbsSpecAug
from ntu_eend.scr.nnet.asr.specaug.specaug import SpecAug
from ntu_eend.scr.task.layers.abs_normalize import AbsNormalize
from ntu_eend.scr.task.layers.global_mvn import GlobalMVN
from ntu_eend.scr.task.layers.utterance_mvn import UtteranceMVN
from ntu_eend.scr.nnet.diar.decoder.abs_decoder import AbsDecoder
from ntu_eend.scr.nnet.diar.decoder.linear_decoder import LinearDecoder
from ntu_eend.scr.nnet.diar.eend_model import EENDModel
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
    default="default",
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
    default="utterance_mvn",
    optional=True,
)
label_aggregator_choices = ClassChoices(
    "label_aggregator",
    classes=dict(label_aggregator=LabelAggregate),
    default="label_aggregator",
)
encoder_choices = ClassChoices(
    "encoder",
    classes=dict(
        transformer=TransformerEncoder,
    ),
    type_check=AbsEncoder,
    default="rnn",
)
decoder_choices = ClassChoices(
    "decoder",
    classes=dict(linear=LinearDecoder),
    type_check=AbsDecoder,
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


class EENDEDATask(AbsTask):
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
        # --encoder and --encoder_conf
        encoder_choices,
        # --decoder and --decoder_conf
        decoder_choices,
        # --label_aggregator and --label_aggregator_conf
        label_aggregator_choices,
        # --attractor and --attractor_conf
        attractor_choices,
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
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of input dimension of the feature",
        )

        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(EENDModel),
            help="The keyword arguments for model class.",
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=True,
            help="Apply preprocessing to data or not",
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
            retval = ("speech", "spk_labels")
        else:
            # Recognition mode
            retval = ("speech",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        # (Note: jiatong): no optional data names for now
        retval = ()
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> EENDModel:
        assert check_argument_types()

        # 1. frontend
        if args.input_size is None:
            # Extract features in the model
            frontend_class = frontend_choices.get_class(args.frontend)
            frontend = frontend_class(**args.frontend_conf)
            input_size = frontend.output_size()
        else:
            # Give features from data-loader
            args.frontend = None
            args.frontend_conf = {}
            frontend = None
            input_size = args.input_size

        # 2. Data augmentation for spectrogram
        if args.specaug is not None:
            specaug_class = specaug_choices.get_class(args.specaug)
            specaug = specaug_class(**args.specaug_conf)
        else:
            specaug = None

        # 3. Normalization layer
        if args.normalize is not None:
            normalize_class = normalize_choices.get_class(args.normalize)
            normalize = normalize_class(**args.normalize_conf)
        else:
            normalize = None

        # 4. Label Aggregator layer
        label_aggregator_class = label_aggregator_choices.get_class(
            args.label_aggregator
        )
        label_aggregator = label_aggregator_class(**args.label_aggregator_conf)

        # 5. Encoder
        encoder_class = encoder_choices.get_class(args.encoder)
        # Note(jiatong): Diarization may not use subsampling when processing
        encoder = encoder_class(input_size=input_size, **args.encoder_conf)

        # 6a. Decoder
        decoder_class = decoder_choices.get_class(args.decoder)
        decoder = decoder_class(
            num_spk=args.num_spk,
            encoder_output_size=encoder.output_size(),
            **args.decoder_conf,
        )

        # 6b. Attractor
        if getattr(args, "attractor", None) is not None:
            attractor_class = attractor_choices.get_class(args.attractor)
            attractor = attractor_class(
                encoder_output_size=encoder.output_size(),
                **args.attractor_conf,
            )
        else:
            attractor = None

        # 7. Build model
        model = EENDModel(
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            label_aggregator=label_aggregator,
            encoder=encoder,
            decoder=decoder,
            attractor=attractor,
            **args.model_conf,
        )

        # FIXME(kamo): Should be done in model?
        # 8. Initialize
        if args.init is not None:
            initialize(model, args.init)

        assert check_return_type(model)
        return model
