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

def peg_insertion_progress(
    env: ManagerBasedRLEnv,
    initial_peg_z: float = 0.025,
    target_insertion: float = 0.010,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Dense insertion reward gated by XY alignment."""

    peg = env.scene[object_cfg.name]

    # Peg position in the environment-local frame.
    peg_pos = (
        peg.data.root_pos_w
        - env.scene.env_origins
    )

    # ---------------------------------------------------------
    # XY alignment
    # ---------------------------------------------------------
    xy_error = torch.norm(peg_pos[:, :2] - torch.tensor(
        [0.5, 0.0],
        device=peg_pos.device,
    ), dim=1)

    # Smooth alignment gate.
  # ~1 when centered, decreases rapidly when misaligned.
    alignment_gate = torch.exp(
        -(xy_error / 0.005) ** 2
    )

    # ---------------------------------------------------------
    # Vertical insertion progress
    # ---------------------------------------------------------
    peg_z = peg_pos[:, 2]

    insertion = initial_peg_z - peg_z

    progress = torch.clamp(
        insertion / target_insertion,
        min=0.0,
        max=1.0,
    )

    # Dense normalized exponential shaping.
    dense_progress = (
        1.0 - torch.exp(-3.0 * progress)
    ) / (
        1.0 - torch.exp(
            torch.tensor(-3.0, device=progress.device)
        )
    )

    insertion_reward = (
        0.5 * progress
        + 0.5 * dense_progress
    )

    # ---------------------------------------------------------
    # M6.6: insertion is rewarded only when aligned.
    # ---------------------------------------------------------
    return insertion_reward * alignment_gate
