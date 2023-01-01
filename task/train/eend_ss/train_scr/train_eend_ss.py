#!/usr/bin/env python3
from .eend_ss import EENDSSTask

def get_parser():
    parser = EENDSSTask.get_parser()
    return parser


def main(cmd=None):
    r"""Diar-Enh training.

    Example:
        % python diar_enh_train.py --print_config --optim adadelta \
                > conf/train.yaml
        % python diar_enh_train.py --config conf/train.yaml
    """
    EENDSSTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
