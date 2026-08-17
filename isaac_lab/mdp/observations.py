from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def peg_position(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Peg position expressed in the robot root frame."""

    robot: Articulation = env.scene["robot"]
    peg: RigidObject = env.scene[object_cfg.name]

    peg_pos_w = peg.data.root_pos_w[:, :3]

    peg_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        peg_pos_w,
    )

    return peg_pos_b


def peg_hole_relative_position(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Peg position relative to the nominal hole center in robot root frame.

    The hole center is defined in the robot/environment local frame at:
        x = 0.5 m
        y = 0.0 m
        z = 0.0 m

    Both peg and hole are therefore represented in the same frame before
    computing the relative displacement.
    """

    peg_pos_b = peg_position(
        env,
        object_cfg=object_cfg,
    )

    hole_pos_b = torch.zeros_like(peg_pos_b)
    hole_pos_b[:, 0] = 0.5

    return peg_pos_b - hole_pos_b
