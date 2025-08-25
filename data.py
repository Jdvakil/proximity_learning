import argparse
import math
import numpy as np
import torch
import random
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Franka robot sinusoidal motion with camera data capture.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets import FRANKA_PANDA_CFG


from scene import Scene

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
        import os
        os.makedirs("./scene_captures", exist_ok=True)
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
            scene = Scene()
            new_sphere_position = scene.generate_random_sphere_position(table_position)
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
    print("[INFO]: Starting main function...")
    try:
        sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1/60.0)  # 60 FPS
        sim = sim_utils.SimulationContext(sim_cfg)
        print("[INFO]: Simulation context created...")
        
        # Set main camera view
        sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
        print("[INFO]: Camera view set...")
        
        # Design scene
        scene = Scene()
        print("[INFO]: Scene object created...")
        scene_entities, _ = scene.design_scene()
        print("[INFO]: Scene designed...")
        sim.reset()
        print("[INFO]: Simulation reset...")
        camera1 = scene_entities["camera1"]
        camera2 = scene_entities["camera2"]
        print("[INFO]: Cameras retrieved...")
        
        print("[INFO]: Setup complete...")
        print("[INFO]: Franka robot will move in sinusoidal motion")
        print("[INFO]: Red sphere placed as fixed obstacle on table surface")
        print("[INFO]: Sphere is kinematic (fixed in place, acts as collision obstacle)")
        print("[INFO]: Camera1 (Wrist view) and Camera2 (Table view) will capture RGB and depth data")
        print("[INFO]: Robot and sphere obstacle data will be printed every few steps")
        print("[INFO]: Sphere obstacle will be repositioned randomly every 1000 simulation steps")
        
        # Run the simulator
        run_simulator(sim, scene_entities)
    except Exception as e:
        print(f"[ERROR]: Exception in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
    simulation_app.close()