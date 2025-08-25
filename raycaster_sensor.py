#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project
# SPDX-License-Identifier: BSD-3-Clause
"""
Franka Panda + ray caster demo.
The end-effector follows a circular trajectory in the robot base frame while the sim runs.
"""

import argparse
import numpy as np

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Raycaster on a Franka Panda with circular EE motion.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Imports after app launch
# -----------------------------------------------------------------------------
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# IK controller + frame math
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms  # ee/base frame math :contentReference[oaicite:1]{index=1}

# Robot config: Franka Panda (Isaac Lab assets)
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip


# -----------------------------------------------------------------------------
# Scene config
# -----------------------------------------------------------------------------
@configclass
class RaycasterSensorSceneCfg(InteractiveSceneCfg):
    """Scene with a Franka Panda and a ray caster mounted on the robot base."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd",
            scale=(1, 1, 1),
        ),
    )

    # light
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # robot
    robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # ray caster attached to the fixed base link
    # ray_caster = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
    #     update_period=1.0 / 60.0,
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
    #     mesh_prim_paths=["/World/Ground"],
    #     attach_yaw_only=True,
    #     pattern_cfg=patterns.LidarPatternCfg(
    #         channels=5, vertical_fov_range=[-20, 20], horizontal_fov_range=[-20, 20], horizontal_res=1.0
    #     ),
    #     debug_vis=not args_cli.headless,
    # )
    # Ray caster attached to Franka end-effector
    ray_caster = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand",         # <-- was panda_link0
        update_period=1.0 / 60.0,
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.12),  # a bit in front of the fingers
            # rot=...               # optional: see note below to tilt it
        ),
        mesh_prim_paths=["/World/Ground"],  # what to raycast against
        attach_yaw_only=False,              # inherit full EE orientation
        pattern_cfg=patterns.LidarPatternCfg(
            channels=5, vertical_fov_range=[-20, 20], horizontal_fov_range=[-20, 20], horizontal_res=1.0
        ),
        debug_vis=not args_cli.headless,
    )


# -----------------------------------------------------------------------------
# Simulation loop
# -----------------------------------------------------------------------------
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator with a circular EE trajectory."""
    device = args_cli.device
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # ---------------------------
    # Setup: find joints/links
    # ---------------------------
    robot = scene["robot"]

    # Arm joints (Franka 7-DOF)
    arm_joint_names = [f"panda_joint{i}" for i in range(1, 8)]
    arm_joint_ids, _ = robot.find_joints(arm_joint_names)

    # End-effector body (common names: panda_hand -> fallback to link8/link7 if needed)
    ee_candidates = ["panda_hand", "panda_link8", "panda_link7"]
    ee_body_idx = None
    for name in ee_candidates:
        ids, _ = robot.find_bodies([name])
        if len(ids) > 0:
            ee_body_idx = ids[0]
            ee_body_name = name
            break
    if ee_body_idx is None:
        raise RuntimeError("Could not find a Franka end-effector body (tried: panda_hand/link8/link7).")

    # PhysX Jacobian index for body frames is (body_index - 1) per docs/examples. :contentReference[oaicite:2]{index=2}
    ee_jacobi_idx = ee_body_idx - 1

    # ---------------------------
    # Differential IK controller
    # ---------------------------
    ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",          # track full pose (pos + quat)
        use_relative_mode=False,      # absolute target in base frame
        ik_method="dls",              # damped least squares is robust
        ik_params={"lambda_val": 0.05},
    )
    diff_ik = DifferentialIKController(ik_cfg, num_envs=scene.num_envs, device=device)  # :contentReference[oaicite:3]{index=3}

    # Circle parameters (in the robot base frame)
    radius = 0.30          # meters
    omega = 0.5            # rad/s
    center_b = None        # (num_envs, 3) set on reset from current EE pose (base frame)
    quat_goal_b = None     # (num_envs, 4) hold orientation fixed while drawing the circle

    # Save-once trigger just kept from the original sample
    triggered = True
    countdown = 42

    # ---------------------------
    # Simulate
    # ---------------------------
    while simulation_app.is_running():
        # Periodic reset
        if count % 500 == 0:
            count = 0
            # reset robot root pose/vel around env origins
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])

            # jitter joints a bit
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.05
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

            # clear buffers
            scene.reset()
            diff_ik.reset()
            print("[INFO] Resetting robot state...")

            # Establish circle center/orientation at current EE pose (in base frame)
            ee_pose_w = robot.data.body_state_w[:, ee_body_idx, 0:7]         # (N, 7): [x,y,z,qw,qx,qy,qz]
            root_pose_w = robot.data.root_state_w[:, 0:7]                    # (N, 7)
            ee_pos_b, ee_quat_b = subtract_frame_transforms(                 # world -> base frame conversion
                root_pose_w[:, 0:3], root_pose_w[:, 3:7],
                ee_pose_w[:, 0:3], ee_pose_w[:, 3:7],
            )
            center_b = ee_pos_b.clone()
            quat_goal_b = ee_quat_b.clone()

        # ------------------------------------
        # Build a circular EE pose command
        # ------------------------------------
        theta = torch.tensor(sim_time * omega, device=device)
        # same circle in every env
        # offset_b = torch.stack([radius * torch.cos(theta), radius * torch.sin(theta), torch.tensor(0.0, device=device)])
        a = 0.15  # X semi-axis (m)
        b = 0.07  # Y semi-axis (m)
        offset_b = torch.stack([
            a * torch.cos(theta),  # X
            b * torch.sin(theta),  # Y
            torch.zeros((), device=device)
        ])
        offset_b = offset_b.unsqueeze(0).repeat(scene.num_envs, 1)  # (N, 3)

        # desired pose in base frame
        pos_cmd_b = center_b + offset_b
        quat_cmd_b = quat_goal_b  # keep orientation constant

        # Pack command for the controller (x,y,z,qw,qx,qy,qz) and send
        ik_commands = torch.zeros(scene.num_envs, diff_ik.action_dim, device=device)
        ik_commands[:, 0:3] = pos_cmd_b
        ik_commands[:, 3:7] = quat_cmd_b
        diff_ik.set_command(ik_commands)  # :contentReference[oaicite:4]{index=4}

        # ------------------------------------
        # Run IK -> joint targets
        # ------------------------------------
        # Quantities from sim: Jacobian, current EE pose (base frame), joint positions
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_joint_ids]  # (N, 6, 7) :contentReference[oaicite:5]{index=5}
        ee_pose_w = robot.data.body_state_w[:, ee_body_idx, 0:7]
        root_pose_w = robot.data.root_state_w[:, 0:7]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7],
        )
        joint_pos = robot.data.joint_pos[:, arm_joint_ids]

        # Compute desired arm joint positions to realize the command pose
        arm_joint_pos_cmd = diff_ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)  # -> (N, 7) :contentReference[oaicite:6]{index=6}

        # Write full joint target vector (preserve non-arm joints if any)
        full_joint_pos_cmd = robot.data.joint_pos.clone()
        full_joint_pos_cmd[:, arm_joint_ids] = arm_joint_pos_cmd
        robot.set_joint_position_target(full_joint_pos_cmd)

        # ------------------------------------
        # Step sim & update scene
        # ------------------------------------
        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)

        # (Optional) Ray caster readout
        print("-------------------------------")
        print(scene["ray_caster"])
        print("Ray cast hit results: ", scene["ray_caster"].data.ray_hits_w)

        # (Optional) one-shot save from sensor (kept from original sample)
        if not triggered:
            if countdown > 0:
                countdown -= 1
                continue
            data = scene["ray_caster"].data.ray_hits_w.cpu().numpy()
            np.save("cast_data.npy", data)
            triggered = True
        else:
            continue


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])

    scene_cfg = RaycasterSensorSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete...")

    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()