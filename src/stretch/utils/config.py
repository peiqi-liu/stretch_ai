# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import json
import os
from pathlib import Path
from typing import Optional, Tuple

import yacs.config
import yaml
from loguru import logger

import stretch


class Config(yacs.config.CfgNode):
    """store a yaml config"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, new_allowed=True)


def get_config(path: str, opts: Optional[list] = None) -> Tuple[Config, str]:
    """Get configuration and ensure consistency between configurations
    inherited from the task and defaults and our code's configuration.

    Arguments:
        path: path to our code's config
        opts: command line arguments overriding the config
    """
    try:
        if os.environ["STRETCHPY_ROOT"]:
            path = os.path.join(os.environ["STRETCHPY_ROOT"], path)
    except KeyError:
        logger.warning("STRETCHPY_ROOT environment variable not set when trying to read configs!")

    # Start with our code's config
    config = Config()
    config.merge_from_file(path)

    # Add command line arguments
    if opts is not None:
        config.merge_from_list(opts)
    config.freeze()

    # Generate a string representation of our code's config
    config_dict = yaml.load(open(path), Loader=yaml.FullLoader)
    if opts is not None:
        for i in range(0, len(opts), 2):
            dict = config_dict
            keys = opts[i].split(".")
            if "TASK_CONFIG" in keys:
                continue
            value = opts[i + 1]
            for key in keys[:-1]:
                dict = dict[key]
            dict[keys[-1]] = value
    config_str = json.dumps(config_dict, indent=4)

    return config, config_str


# New configuration system
CONTROL_CONFIG_DIR = str(Path(stretch.__path__[0]).parent.resolve() / "config" / "control")


def load_config(visualize: bool = False, print_images: bool = True, config_path=None, **kwargs):
    """Load config path for real world experiments and use proper presets. This is based on the version from HomeRobot, and is designed therefore to work with Habitat."""
    config, config_str = get_config(config_path)
    config.defrost()
    config.NUM_ENVIRONMENTS = 1
    config.VISUALIZE = int(visualize)
    config.PRINT_IMAGES = int(print_images)
    config.EXP_NAME = "debug"
    if config.GROUND_TRUTH_SEMANTICS != 0:
        raise RuntimeError("No ground truth semantics in the real world!")
    config.freeze()
    return config
