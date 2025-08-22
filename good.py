#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates a Franka robot moving in sinusoidal motion with camera data capture
and a randomly placed sphere on the table.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/franka_sinusoidal_cameras.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import numpy as np
import torch
import random

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Franka robot sinusoidal motion with camera data capture.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Import Franka configuration
from isaaclab_assets import FRANKA_PANDA_CFG


def generate_random_sphere_position(table_center, table_size=(0.8, 0.8), sphere_radius=0.05):
    """
    Generate a random position for a sphere on the table surface.
    
    Args:
        table_center: (x, y, z) position of table center
        table_size: (width, depth) of table surface 
        sphere_radius: radius of sphere to ensure it stays on table
        
    Returns:
        (x, y, z) position for sphere
    """
    # Table dimensions (assuming standard table size)
    table_x, table_y, table_z = table_center
    table_width, table_depth = table_size
    
    # Generate random x, y within table bounds (with margin for sphere radius)
    margin = sphere_radius + 0.05  # Small additional margin
    random_x = table_x + random.uniform(-table_width/2 + margin, table_width/2 - margin)
    random_y = table_y + random.uniform(-table_depth/2 + margin, table_depth/2 - margin)
    
    # Place sphere on table surface (table_z is table center, add half table height + sphere radius)
    sphere_z = table_z + 0.02 + sphere_radius  # 0.02 is approximate table half-height
    
    return (random_x, random_y, sphere_z)


def design_scene() -> tuple[dict, dict]:
    """Designs the scene with Franka robot, cameras, and a random sphere."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create origin for robot
    prim_utils.create_prim("/World/Origin1", "Xform", translation=[0.0, 0.0, 0.0])
    
    # Table
    table_position = (0.55, 0.0, 1.05)
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
    cfg.func("/World/Origin1/Table", cfg, translation=table_position)
    
    # Generate random sphere position on table
    sphere_position = generate_random_sphere_position(table_position)
    print(f"[INFO]: Placing sphere at position: {sphere_position}")
    
    # Create sphere as fixed obstacle (no physics, static collision)
    sphere_cfg = RigidObjectCfg(
        prim_path="/World/Origin1/Sphere",
        spawn=sim_utils.SphereCfg(
            radius=0.05,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # Make it kinematic (not affected by forces)
                disable_gravity=True,    # Disable gravity
                solver_position_iteration_count=0,
                solver_velocity_iteration_count=0,
                max_angular_velocity=0.0,
                max_linear_velocity=0.0,
                max_depenetration_velocity=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),  # Mass doesn't matter for kinematic
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,  # Enable collision detection
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),  # Red sphere for visibility
                metallic=0.2,
                roughness=0.5,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=sphere_position,
            rot=(1.0, 0.0, 0.0, 0.0),  # No rotation
        ),
    )
    sphere = RigidObject(cfg=sphere_cfg)
    
    # Franka Robot
    franka_arm_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/Origin1/Robot")
    franka_arm_cfg.init_state.pos = (0.0, 0.0, 1.05)
    franka_panda = Articulation(cfg=franka_arm_cfg)

    # Camera 1 - Wrist camera (mounted on robot hand)
    camera1_cfg = CameraCfg(
        prim_path="/World/Origin1/Robot/panda_hand/wrist_cam",
        update_period=0.1,  # 10 FPS
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, 
            focus_distance=400.0, 
            horizontal_aperture=20.955, 
            clipping_range=(0.1, 2.0)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.13, 0.0, -0.15), 
            rot=(-0.70614, 0.03701, 0.03701, -0.70614), 
            convention="ros"
        ),
    )
    camera1 = Camera(cfg=camera1_cfg)

    # Camera 2 - Table view camera (external fixed camera)
    camera2_cfg = CameraCfg(
        prim_path="/World/table_cam",
        update_period=0.1,  # 10 FPS
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, 
            focus_distance=400.0, 
            horizontal_aperture=20.955, 
            clipping_range=(0.1, 2.0)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(2.0, 0.0, 1.6), 
            # rot=(0.35355, -0.61237, -0.61237, 0.35355),
            rot = (0.56099, 0.43046, 0.43046, 0.56099),  # Adjusted for better view
            convention="opengl"
        ),
    )
    camera2 = Camera(cfg=camera2_cfg)

    # Return scene entities
    scene_entities = {
        "franka_panda": franka_panda,
        "sphere": sphere,
        "camera1": camera1,
        "camera2": camera2,
    }
    
    return scene_entities, {}


def apply_sinusoidal_motion(robot: Articulation, sim_time: float, amplitude: float = 0.3, frequency: float = 0.5):
    """Apply sinusoidal motion to robot joints."""
    # Get current joint positions
    current_joint_pos = robot.data.default_joint_pos.clone()
    
    # Apply sinusoidal motion to different joints with different phases
    joint_offsets = torch.zeros_like(current_joint_pos)
    
    # Joint 1: Shoulder Pan
    joint_offsets[:, 0] = amplitude * math.sin(2 * math.pi * frequency * sim_time)
    
    # Joint 2: Shoulder Lift  
    joint_offsets[:, 1] = amplitude * 0.5 * math.sin(2 * math.pi * frequency * sim_time + math.pi/4)
    
    # Joint 3: Elbow
    joint_offsets[:, 2] = amplitude * 0.7 * math.sin(2 * math.pi * frequency * sim_time + math.pi/2)
    
    # Joint 4: Wrist 1
    joint_offsets[:, 3] = amplitude * 0.4 * math.sin(2 * math.pi * frequency * sim_time + 3*math.pi/4)
    
    # Joint 5: Wrist 2
    joint_offsets[:, 4] = amplitude * 0.3 * math.sin(2 * math.pi * frequency * sim_time + math.pi)
    
    # Joint 6: Wrist 3
    joint_offsets[:, 5] = amplitude * 0.2 * math.sin(2 * math.pi * frequency * sim_time + 5*math.pi/4)
    
    # Joint 7: Wrist Rotate
    joint_offsets[:, 6] = amplitude * 0.1 * math.sin(2 * math.pi * frequency * sim_time + 3*math.pi/2)
    
    # Calculate target positions
    joint_pos_target = current_joint_pos + joint_offsets
    
    # Clamp to joint limits
    joint_pos_target = joint_pos_target.clamp_(
        robot.data.soft_joint_pos_limits[..., 0], 
        robot.data.soft_joint_pos_limits[..., 1]
    )
    
    return joint_pos_target


def print_robot_data(robot: Articulation, sim_time: float):
    """Print robot state data."""
    print(f"\n--- Robot Data at t={sim_time:.2f}s ---")
    print(f"Joint Positions: {robot.data.joint_pos[0].cpu().numpy()}")
    print(f"Joint Velocities: {robot.data.joint_vel[0].cpu().numpy()}")
    print(f"End Effector Position: {robot.data.root_pos_w[0].cpu().numpy()}")
    print(f"End Effector Orientation: {robot.data.root_quat_w[0].cpu().numpy()}")


def print_sphere_data(sphere: RigidObject, sim_time: float):
    """Print sphere state data (position only since it's kinematic)."""
    print(f"\n--- Sphere Obstacle Data at t={sim_time:.2f}s ---")
    print(f"Sphere Position (Fixed): {sphere.data.root_pos_w[0].cpu().numpy()}")
    print(f"Sphere Orientation (Fixed): {sphere.data.root_quat_w[0].cpu().numpy()}")
    # Note: Velocities should be zero for kinematic objects


def save_camera_data(camera: Camera, camera_name: str, sim_time: float):
    """Save camera RGB and depth data."""
    if camera.data.output is not None:
        if "rgb" in camera.data.output:
            rgb_data = camera.data.output["rgb"][0].cpu().numpy()  # Get first environment
            print(f"--- {camera_name} RGB Data at t={sim_time:.2f}s ---")
            print(f"RGB Image Shape: {rgb_data.shape}")
            print(f"RGB Image Range: [{rgb_data.min():.3f}, {rgb_data.max():.3f}]")
        
        if "distance_to_image_plane" in camera.data.output:
            depth_data = camera.data.output["distance_to_image_plane"][0].cpu().numpy()
            print(f"--- {camera_name} Depth Data at t={sim_time:.2f}s ---")
            print(f"Depth Image Shape: {depth_data.shape}")
            print(f"Depth Range: [{depth_data.min():.3f}, {depth_data.max():.3f}] meters")
        
        # Optional: Save images to file (uncomment if needed)
        import cv2
        if "rgb" in camera.data.output:
            # rgb_uint8 = rgb_data[:,:,::-1] #(rgb_data * 255).astype(np.uint8)
            cv2.imwrite(f"./scene_captures/{camera_name}_rgb_t{sim_time:.2f}.png", cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR))
        if "distance_to_image_plane" in camera.data.output:
            depth_normalized = (depth_data / depth_data.max() * 255).astype(np.uint8)
            cv2.imwrite(f"./scene_captures/{camera_name}_depth_t{sim_time:.2f}.png", depth_normalized)


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, object]):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    data_capture_interval = 50  # Capture data every 50 steps
    
    # Get entities
    robot = entities["franka_panda"]
    sphere = entities["sphere"]
    camera1 = entities["camera1"]
    camera2 = entities["camera2"]
    
    print("[INFO]: Starting sinusoidal motion simulation...")
    
    # Simulate physics
    while simulation_app.is_running():
        # Reset periodically to prevent drift
        if count % 1000 == 0:
            print(f"[INFO]: Resetting robot state at step {count}...")
            # Reset robot state
            root_state = robot.data.default_root_state.clone()
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            
            # Reset joint positions
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            
            # Reset sphere to new random position (kinematic object)
            table_position = (0.55, 0.0, 1.05)
            new_sphere_position = generate_random_sphere_position(table_position)
            print(f"[INFO]: Moving sphere obstacle to new random position: {new_sphere_position}")
            
            # For kinematic objects, we only need to set position
            sphere_root_state = sphere.data.default_root_state.clone()
            sphere_root_state[:, :3] = torch.tensor(new_sphere_position, device=sphere.device).unsqueeze(0)
            sphere_root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=sphere.device).unsqueeze(0)  # Reset rotation
            sphere_root_state[:, 7:] = 0.0  # Zero velocities (should stay zero for kinematic)
            sphere.write_root_pose_to_sim(sphere_root_state[:, :7])
            sphere.write_root_velocity_to_sim(sphere_root_state[:, 7:])
            sphere.reset()
            
            # Reset simulation time for motion pattern
            sim_time = 0.0
        
        # Apply sinusoidal motion to robot
        joint_pos_target = apply_sinusoidal_motion(robot, sim_time, amplitude=0.4, frequency=0.3)
        robot.set_joint_position_target(joint_pos_target)
        robot.write_data_to_sim()
        
        # Perform simulation step
        sim.step()
        
        # Update entities
        robot.update(sim_dt)
        sphere.update(sim_dt)
        camera1.update(sim_dt)
        camera2.update(sim_dt)
        
        # Capture and print data at intervals
        if count % data_capture_interval == 0:
            # Print robot data
            print_robot_data(robot, sim_time)
            
            # Print sphere data
            print_sphere_data(sphere, sim_time)
            
            # Save camera data
            save_camera_data(camera1, "WristCam", sim_time)
            save_camera_data(camera2, "TableCam", sim_time)
            
            print("-" * 60)
        
        # Update time and counters
        sim_time += sim_dt
        count += 1
        
        # Optional: Stop after certain time for testing
        if sim_time > 30.0:  # Run for 30 seconds
            print("[INFO]: Simulation complete!")
            break


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1/60.0)  # 60 FPS
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # Set main camera view
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    
    # Design scene
    scene_entities, _ = design_scene()
    
    # Play the simulator
    sim.reset()
    
    # Cameras are now configured with offsets, no need to manually set poses
    camera1 = scene_entities["camera1"]
    camera2 = scene_entities["camera2"]
    
    print("[INFO]: Setup complete...")
    print("[INFO]: Franka robot will move in sinusoidal motion")
    print("[INFO]: Red sphere placed as fixed obstacle on table surface")
    print("[INFO]: Sphere is kinematic (fixed in place, acts as collision obstacle)")
    print("[INFO]: Camera1 (Wrist view) and Camera2 (Table view) will capture RGB and depth data")
    print("[INFO]: Robot and sphere obstacle data will be printed every few steps")
    print("[INFO]: Sphere obstacle will be repositioned randomly every 1000 simulation steps")
    
    # Run the simulator
    run_simulator(sim, scene_entities)


if __name__ == "__main__":
    # Run the main function
    main()
    # Close sim app
    simulation_app.close()