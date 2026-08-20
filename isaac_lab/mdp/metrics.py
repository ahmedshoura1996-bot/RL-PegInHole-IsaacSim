from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .observations import peg_hole_relative_position

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def peg_hole_success(
    env: ManagerBasedRLEnv,
    xy_tolerance: float = 0.0005,
    insertion_depth: float = 0.010,
    initial_peg_z: float = 0.025,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Binary Peg-in-Hole task success metric.

    Success requires both:
    1. Peg XY error within the nominal radial clearance.
    2. Peg insertion depth reaching the target depth.

    This metric is intended for evaluation and does not terminate
    the environment.
    """

    relative_pos = peg_hole_relative_position(
        env,
        object_cfg=object_cfg,
    )

    xy_error = torch.norm(relative_pos[:, :2], dim=1)

    peg = env.scene[object_cfg.name]

    peg_z = (
        peg.data.root_pos_w[:, 2]
        - env.scene.env_origins[:, 2]
    )

    insertion = initial_peg_z - peg_z

    xy_ok = xy_error <= xy_tolerance
    insertion_ok = insertion >= insertion_depth

    return (xy_ok & insertion_ok).float()
