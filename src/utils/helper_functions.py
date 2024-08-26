import os

import auglib
import torch
from omegaconf import DictConfig


def set_device(cfg: DictConfig) -> None:
    """
    Checks if cuda as available on the device and safe the used device in the config
    @param cfg: Dict Config
    @return:
    """
    torch.manual_seed(1337)
    auglib.seed(1337)

    if torch.cuda.is_available():
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.cuda_id)
        cfg.device = 'cuda:' + str(cfg.cuda_id)
    else:
        cfg.device = 'cpu'
