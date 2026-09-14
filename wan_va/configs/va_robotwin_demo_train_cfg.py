import os

from easydict import EasyDict

from .va_robotwin_train_cfg import va_robotwin_train_cfg


va_robotwin_demo_train_cfg = EasyDict(
    __name__='Config: VA RobotWin demonstration paired post-training')
va_robotwin_demo_train_cfg.update(va_robotwin_train_cfg)
va_robotwin_demo_train_cfg.enable_demo_conditioning = True
va_robotwin_demo_train_cfg.demonstration_manifest_path = os.environ.get(
    'NEXT_FORCING_DEMONSTRATION_MANIFEST')
va_robotwin_demo_train_cfg.demonstration_split = os.environ.get(
    'NEXT_FORCING_DEMONSTRATION_SPLIT', 'train')
