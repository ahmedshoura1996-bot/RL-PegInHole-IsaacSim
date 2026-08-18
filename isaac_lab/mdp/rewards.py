from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .observations import peg_hole_relative_position

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def peg_hole_xy_alignment(
    env: ManagerBasedRLEnv,
    std: float = 0.01,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward XY alignment of the peg with the nominal hole center.

    The reward is based only on the horizontal displacement between the
    peg center and the nominal hole center, expressed in the robot root
    frame.

    A perfectly centered peg receives a reward of 1.0.
    """
    relative_pos = peg_hole_relative_position(
        env,
        object_cfg=object_cfg,
    )

    xy_error = torch.norm(relative_pos[:, :2], dim=1)

    return 1.0 - torch.tanh(xy_error / std)
