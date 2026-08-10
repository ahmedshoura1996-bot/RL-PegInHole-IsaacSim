"""
Peg-in-Hole Environment Configuration

First milestone:
Franka Panda + IK Absolute + Cylindrical Peg.

This configuration is intentionally kept small for the first
Isaac Lab smoke test. Hole geometry, observations, rewards,
contact sensing, and termination logic will be added in later
milestones.
"""

from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.shapes.shapes_cfg import CylinderCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, CollisionPropertiesCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift.config.franka.ik_abs_env_cfg import (
    FrankaCubeLiftEnvCfg as FrankaIKLiftEnvCfg,
)


# ---------------------------------------------------------------------------
# Temporary geometry parameters for the first smoke test.
# These are NOT the final experimental dimensions.
# ---------------------------------------------------------------------------

TEST_PEG_RADIUS = 0.01   # 10 mm radius -> 20 mm diameter
TEST_PEG_HEIGHT = 0.05   # 50 mm
@configclass
class PegInHoleEnvCfg(FrankaIKLiftEnvCfg):
    """First Peg-in-Hole environment milestone."""

    def __post_init__(self):
        # Initialize the official Franka IK-Absolute Lift configuration.
        super().__post_init__()

        # Replace the cube with a simple cylindrical rigid peg.
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.5, 0.0, TEST_PEG_HEIGHT / 2.0),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            spawn=CylinderCfg(
                radius=TEST_PEG_RADIUS,
                height=TEST_PEG_HEIGHT,
                axis="Z",
                rigid_props=RigidBodyPropertiesCfg(
                    disable_gravity=False,
                ),
                collision_props=CollisionPropertiesCfg(
                    collision_enabled=True,
                ),
            ),
        )
