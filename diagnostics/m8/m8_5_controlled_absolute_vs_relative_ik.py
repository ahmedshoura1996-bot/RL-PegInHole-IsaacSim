import torch

from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.utils.math import compute_pose_error


def print_pose(label, pos, quat):
    print(f"\n{label}")
    print(f"  pos  = {pos.detach().cpu().numpy()}")
    print(f"  quat = {quat.detach().cpu().numpy()}")
    print(f"  quat_norm = {torch.linalg.norm(quat, dim=-1).detach().cpu().numpy()}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_envs = 1
    num_joints = 7

    print("=" * 80)
    print("M8.5 - CONTROLLED ABSOLUTE vs RELATIVE DIFFERENTIAL IK TEST")
    print("=" * 80)
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Controlled current robot state
    # ------------------------------------------------------------------
    ee_pos = torch.tensor(
        [[0.35, 0.00, 0.45]],
        dtype=torch.float32,
        device=device,
    )

    # Valid normalized quaternion in Isaac Lab convention: (w, x, y, z)
    ee_quat = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
        device=device,
    )

    # Small controlled Cartesian displacement.
    delta_xyz = torch.tensor(
        [[0.02, -0.01, 0.015]],
        dtype=torch.float32,
        device=device,
    )

    # Small controlled orientation delta:
    # [droll, dpitch, dyaw]
    delta_rot = torch.tensor(
        [[0.0, 0.0, 0.05]],
        dtype=torch.float32,
        device=device,
    )

    delta_pose = torch.cat((delta_xyz, delta_rot), dim=1)

    # ------------------------------------------------------------------
    # Create the expected target using the same math used by relative IK.
    # ------------------------------------------------------------------
    from isaaclab.utils.math import apply_delta_pose

    expected_rel_pos, expected_rel_quat = apply_delta_pose(
        ee_pos,
        ee_quat,
        delta_pose,
    )

    print_pose("CURRENT EE POSE", ee_pos, ee_quat)

    print("\nCONTROLLED DELTA")
    print(f"  delta_xyz = {delta_xyz.cpu().numpy()}")
    print(f"  delta_rot = {delta_rot.cpu().numpy()}")

    print_pose(
        "EXPECTED TARGET FROM RELATIVE DELTA",
        expected_rel_pos,
        expected_rel_quat,
    )

    # ------------------------------------------------------------------
    # Absolute controller
    # ------------------------------------------------------------------
    abs_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": 0.01},
    )

    abs_controller = DifferentialIKController(
        cfg=abs_cfg,
        num_envs=num_envs,
        device=device,
    )

    abs_controller.set_command(
         torch.cat((expected_rel_pos, expected_rel_quat), dim=1),
        ee_pos,
        ee_quat,
    )

    # ------------------------------------------------------------------
    # Relative controller
    # ------------------------------------------------------------------
    rel_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=True,
        ik_method="dls",
        ik_params={"lambda_val": 0.01},
    )

    rel_controller = DifferentialIKController(
        cfg=rel_cfg,
        num_envs=num_envs,
        device=device,
    )

    rel_controller.set_command(
        delta_pose,
        ee_pos,
        ee_quat,
    )

    # ------------------------------------------------------------------
    # Compare controller-generated targets
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CONTROLLER TARGET COMPARISON")
    print("=" * 80)

    print_pose(
        "ABSOLUTE CONTROLLER TARGET",
         abs_controller.ee_pos_des,
        abs_controller.ee_quat_des,
    )

    print_pose(
        "RELATIVE CONTROLLER TARGET",
        rel_controller.ee_pos_des,
        rel_controller.ee_quat_des,
    )

    pos_diff = torch.linalg.norm(
        abs_controller.ee_pos_des - rel_controller.ee_pos_des,
        dim=1,
    )

    quat_diff = torch.linalg.norm(
        abs_controller.ee_quat_des - rel_controller.ee_quat_des,
        dim=1,
    )

    print("\nTARGET DIFFERENCE")
    print(f"  position difference = {pos_diff.item():.10f} m")
    print(f"  quaternion difference = {quat_diff.item():.10f}")

    # ------------------------------------------------------------------
    # Controlled Jacobian and joint state
    #
    # This is deliberately synthetic and well-conditioned.
    # The goal here is to verify controller semantics, not robot dynamics.
    # ------------------------------------------------------------------
    torch.manual_seed(42)

    jacobian = torch.tensor(
        [
            [
   [0.10, 0.00, 0.00, 0.20, 0.00, 0.00, 0.10],
                [0.00, 0.10, 0.00, 0.00, 0.20, 0.00, 0.10],
                [0.00, 0.00, 0.10, 0.00, 0.00, 0.20, 0.10],
                [0.20, 0.00, 0.00, 0.10, 0.00, 0.00, 0.05],
                [0.00, 0.20, 0.00, 0.00, 0.10, 0.00, 0.05],
                [0.00, 0.00, 0.20, 0.00, 0.00, 0.10, 0.05],
            ]
        ],
        dtype=torch.float32,
        device=device,
    )

    joint_pos = torch.tensor(
        [[0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.5]],
        dtype=torch.float32,
        device=device,
    )

    # ------------------------------------------------------------------
    # Compute IK joint targets
    # ------------------------------------------------------------------
    abs_joint_target = abs_controller.compute(
        ee_pos,
        ee_quat,
        jacobian,
        joint_pos,
    )

    rel_joint_target = rel_controller.compute(
        ee_pos,
        ee_quat,
        jacobian,
        joint_pos,
    )

    print("\n" + "=" * 80)
    print("JOINT TARGET COMPARISON")
    print("=" * 80)

    print(
        "ABS joint target:\n",
        abs_joint_target.detach().cpu().numpy(),
    )

    print(
        "REL joint target:\n",
        rel_joint_target.detach().cpu().numpy(),
    )

    joint_diff = torch.linalg.norm(
        abs_joint_target - rel_joint_target,
        dim=1,
    )

    print(
        f"\nJoint-target difference = {joint_diff.item():.10f} rad"
    )

    # ------------------------------------------------------------------
    # Verify pose errors
    # ------------------------------------------------------------------
    abs_pos_error, abs_rot_error = compute_pose_error(
        ee_pos,
        ee_quat,
        abs_controller.ee_pos_des,
        abs_controller.ee_quat_des,
        rot_error_type="axis_angle",
    )

    rel_pos_error, rel_rot_error = compute_pose_error(
        ee_pos,
        ee_quat,
        rel_controller.ee_pos_des,
        rel_controller.ee_quat_des,
         rot_error_type="axis_angle",
    )

    print("\n" + "=" * 80)
    print("POSE ERROR")
    print("=" * 80)

    print(
        "Absolute position error:",
        abs_pos_error.detach().cpu().numpy(),
    )

    print(
        "Relative position error:",
        rel_pos_error.detach().cpu().numpy(),
    )

    print(
        "Absolute rotation error:",
        abs_rot_error.detach().cpu().numpy(),
    )

    print(
        "Relative rotation error:",
        rel_rot_error.detach().cpu().numpy(),
    )

    # ------------------------------------------------------------------
    # Final verdict
    # ------------------------------------------------------------------

    target_match = (
        pos_diff.item() < 1e-6
        and quat_diff.item() < 1e-6
    )

    joint_match = joint_diff.item() < 1e-5

    print("\n" + "=" * 80)
    print("M8.5 VERDICT")
    print("=" * 80)

    if target_match:
        print("PASS: Absolute and Relative controllers generate the same target pose.")
    else:
        print("FAIL: Absolute and Relative target poses do not match.")

    if joint_match:
        print("PASS: Absolute and Relative IK produce the same joint target.")
    else:
        print("FAIL: Absolute and Relative IK produce different joint targets.")

    if target_match and joint_match:
        print("\nFINAL RESULT: M8.5 PASS")
        print("The Differential IK controller semantics are consistent.")
        print("Remaining mismatch should be investigated in:")
        print("  - policy/action representation")
        print("  - action scaling")
        print("  - normalization")
        print("  - PPO training")

    else:
        print("\nFINAL RESULT: M8.5 FAIL")
        print("Investigate IK command interpretation before changing PPO.")

    print("=" * 80)


if __name__ == "__main__":
    main()
