import argparse
import math
import numpy as np
import torch
import random
import os
import pickle
from datetime import datetime
import cv2  # For video recording

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
from data_collector import DataCollector


def get_static_joint_positions(robot: Articulation):
    """Get predefined static start and goal joint positions."""
    # Define a safe start position (slightly bent arm configuration)
    start_joint_pos = torch.zeros_like(robot.data.default_joint_pos)
    start_joint_pos[:, 0] = -0.13935425877571106      # Shoulder pan
    start_joint_pos[:, 1] = -0.020481698215007782   # Shoulder lift
    start_joint_pos[:, 2] = -0.05201413854956627     # Elbow
    start_joint_pos[:, 3] = -2.0691256523132324     # Elbow flex
    start_joint_pos[:, 4] = 0.05058913677930832      # Wrist 1
    start_joint_pos[:, 5] = 2.0028650760650635      # Wrist 2
    start_joint_pos[:, 6] = -0.9167874455451965      # Wrist 3
    start_joint_pos[:, 7:] = robot.data.default_joint_pos[:, 7:]  # Keep fingers default
    
    # Define a different goal position (extended arm configuration)
    goal_joint_pos = torch.zeros_like(robot.data.default_joint_pos)
    goal_joint_pos[:, 0] = 1.2       # Shoulder pan
    goal_joint_pos[:, 1] = -0.2      # Shoulder lift
    goal_joint_pos[:, 2] = -0.8      # Elbow
    goal_joint_pos[:, 3] = -1.5      # Elbow flex
    goal_joint_pos[:, 4] = 0.5       # Wrist 1
    goal_joint_pos[:, 5] = 1.0       # Wrist 2
    goal_joint_pos[:, 6] = -0.5      # Wrist 3
    goal_joint_pos[:, 7:] = robot.data.default_joint_pos[:, 7:]  # Keep fingers default
    
    return start_joint_pos, goal_joint_pos

def generate_random_joint_position(robot: Articulation, avoid_limits: bool = True):
    """Generate a random valid joint position for the robot."""
    if avoid_limits:
        # Stay well within joint limits for safer motion
        joint_limits_lower = robot.data.soft_joint_pos_limits[..., 0] * 0.8
        joint_limits_upper = robot.data.soft_joint_pos_limits[..., 1] * 0.8
    else:
        joint_limits_lower = robot.data.soft_joint_pos_limits[..., 0]
        joint_limits_upper = robot.data.soft_joint_pos_limits[..., 1]
    
    # Generate random joint positions within limits
    random_joints = torch.zeros_like(robot.data.default_joint_pos)
    for i in range(7):  # First 7 joints (arm only, not fingers)
        # Generate random value between 0 and 1, then scale to joint limits
        random_val = torch.rand(1, device=robot.device)
        random_joints[:, i] = random_val * (joint_limits_upper[:, i] - joint_limits_lower[:, i]) + joint_limits_lower[:, i]
    
    # Keep finger joints at default (closed)
    random_joints[:, 7:] = robot.data.default_joint_pos[:, 7:]
    
    return random_joints

def interpolate_joint_trajectory(start_pos: torch.Tensor, goal_pos: torch.Tensor, num_steps: int):
    """Create a linear interpolation trajectory between start and goal positions."""
    trajectory = []
    for step in range(num_steps):
        alpha = step / (num_steps - 1)  # Linear interpolation factor
        interpolated_pos = (1 - alpha) * start_pos + alpha * goal_pos
        trajectory.append(interpolated_pos)
    return trajectory

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

def print_robot_data(robot: Articulation, sim_time: float, data_collector: DataCollector = None, step_count: int = 0):
    """Print robot state data and optionally collect it for saving."""
    print(f"\n--- Robot Data at t={sim_time:.2f}s ---")
    print(f"Joint Positions: {robot.data.joint_pos[0].cpu().numpy()}")
    print(f"Joint Velocities: {robot.data.joint_vel[0].cpu().numpy()}")
    print(f"End Effector Position: {robot.data.root_pos_w[0].cpu().numpy()}")
    print(f"End Effector Orientation: {robot.data.root_quat_w[0].cpu().numpy()}")
    
    # Collect data for pickle files if collector is provided
    if data_collector is not None:
        data_collector.collect_robot_data(robot, sim_time, step_count)


def print_sphere_data(sphere: RigidObject, sim_time: float, data_collector: DataCollector = None, step_count: int = 0):
    """Print sphere state data and optionally collect it for saving."""
    print(f"\n--- Sphere Obstacle Data at t={sim_time:.2f}s ---")
    print(f"Sphere Position (Fixed): {sphere.data.root_pos_w[0].cpu().numpy()}")
    print(f"Sphere Orientation (Fixed): {sphere.data.root_quat_w[0].cpu().numpy()}")
    # Note: Velocities should be zero for kinematic objects
    
    # Collect sphere data for pickle files if collector is provided
    if data_collector is not None:
        data_collector.collect_sphere_data(sphere, sim_time, step_count)
def save_camera_data(camera: Camera, camera_name: str, sim_time: float, data_collector: DataCollector = None, step_count: int = 0):
    """Save camera RGB and depth data and optionally collect metadata."""
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
    
    # Collect camera metadata for pickle files if collector is provided
    if data_collector is not None:
        data_collector.collect_camera_data(camera, camera_name, sim_time, step_count)

def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, object]):
    """Runs the simulation loop with start-to-goal motion episodes."""
    # Define simulation parameters
    sim_dt = sim.get_physics_dt()
    episode_length = 2000  # Number of steps per episode (increased from 50)
    max_episodes = 200   # Total number of episodes to collect
    
    # Get entities
    robot = entities["franka_panda"]
    sphere = entities["sphere"]
    cameras = {"camera1": entities["camera1"], "camera2": entities["camera2"]}
    
    # Initialize data collector for episode-based data collection
    data_collector = DataCollector()
    data_collector.set_metadata(episode_length, sim_dt)
    
    print("[INFO]: Starting start-to-goal motion data collection...")
    print(f"[INFO]: Episodes: {max_episodes}, Steps per episode: {episode_length}")
    print(f"[INFO]: Data will be saved to pickle files in: {data_collector.data_dir}")
    
    # Create directory for video captures
    video_dir = "./scene_captures"
    os.makedirs(video_dir, exist_ok=True)
    
    episode_id = 0
    
    # Get static start and goal positions (same for all episodes)
    static_start_pos, static_goal_pos = get_static_joint_positions(robot)
    print(f"[INFO]: Using static positions for all episodes:")
    print(f"[INFO]: Static start joints: {static_start_pos[0, :7].cpu().numpy()}")
    print(f"[INFO]: Static goal joints: {static_goal_pos[0, :7].cpu().numpy()}")
    
    while simulation_app.is_running() and episode_id < max_episodes:
        # Use the same static start and goal positions for all episodes
        start_joint_pos = static_start_pos.clone()
        goal_joint_pos = static_goal_pos.clone()
        
        # Generate trajectory from start to goal
        trajectory = interpolate_joint_trajectory(start_joint_pos, goal_joint_pos, episode_length)
        
        # Reset robot to start position
        robot.write_joint_state_to_sim(start_joint_pos, torch.zeros_like(start_joint_pos))
        robot.reset()
        
        # Reset sphere to new random position for this episode
        table_position = (0.55, 0.0, 1.05)
        scene = Scene()
        new_sphere_position = scene.generate_random_sphere_position(table_position)
        
        sphere_root_state = sphere.data.default_root_state.clone()
        sphere_root_state[:, :3] = torch.tensor(new_sphere_position, device=sphere.device).unsqueeze(0)
        sphere_root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=sphere.device).unsqueeze(0)
        sphere_root_state[:, 7:] = 0.0
        sphere.write_root_pose_to_sim(sphere_root_state[:, :7])
        sphere.write_root_velocity_to_sim(sphere_root_state[:, 7:])
        sphere.reset()
        
        # Start new episode in data collector
        data_collector.start_new_episode(episode_id, start_joint_pos, goal_joint_pos)
        
        print(f"\n[INFO]: Starting Episode {episode_id + 1}/{max_episodes}")
        print(f"[INFO]: Start joints: {start_joint_pos[0, :7].cpu().numpy()}")
        print(f"[INFO]: Goal joints: {goal_joint_pos[0, :7].cpu().numpy()}")
        print(f"[INFO]: Sphere position: {new_sphere_position}")
        
        # Initialize video recording for this episode
        video_filename = os.path.join(video_dir, f"episode_{episode_id:04d}_tablecam.mp4")
        video_writer = None
        video_frames = []
        
        # Execute the trajectory for this episode
        episode_sim_time = 0.0
        for step_in_episode in range(episode_length):
            # Set target joint position from trajectory
            target_joint_pos = trajectory[step_in_episode]
            robot.set_joint_position_target(target_joint_pos)
            robot.write_data_to_sim()
            
            # Perform simulation step
            sim.step()
            
            # Update entities
            robot.update(sim_dt)
            sphere.update(sim_dt)
            for camera in cameras.values():
                camera.update(sim_dt)
            
            # Capture frame from table camera for video
            table_camera = cameras["camera2"]  # camera2 is the table camera
            if table_camera.data.output is not None and "rgb" in table_camera.data.output:
                rgb_data = table_camera.data.output["rgb"][0].cpu().numpy()
                # Convert from [0,1] float to [0,255] uint8 and from RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor((rgb_data * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                video_frames.append(frame_bgr)
                
                # Initialize video writer on first frame
                if video_writer is None:
                    height, width, _ = frame_bgr.shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    # Record at ~30 FPS (assuming sim runs at 60 FPS, we capture every frame)
                    fps = 30.0
                    video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
            
            # Collect data for this step
            data_collector.collect_step_data(robot, sphere, cameras, episode_sim_time, step_in_episode)
            
            # Optional: Print progress
            if step_in_episode % 25 == 0:  # Print every 25 steps for 200-step episodes
                current_joint_pos = robot.data.joint_pos[0, :7].cpu().numpy()
                goal_joint_pos_np = goal_joint_pos[0, :7].cpu().numpy()
                distance_to_goal = np.linalg.norm(current_joint_pos - goal_joint_pos_np)
                print(f"  Step {step_in_episode}/{episode_length}: Distance to goal: {distance_to_goal:.4f}")
            
            episode_sim_time += sim_dt
        
        # Check if episode was successful (close to goal)
        final_joint_pos = robot.data.joint_pos[0, :7].cpu().numpy()
        goal_joint_pos_np = goal_joint_pos[0, :7].cpu().numpy()
        final_distance = np.linalg.norm(final_joint_pos - goal_joint_pos_np)
        success = final_distance < 0.1  # Success threshold
        
        # Save video for this episode
        if video_writer is not None and len(video_frames) > 0:
            # Write all frames to video
            for frame in video_frames:
                video_writer.write(frame)
            video_writer.release()
            print(f"[INFO]: Saved episode video: {video_filename}")
        elif len(video_frames) > 0:
            # Fallback: save using cv2 directly if video writer failed
            try:
                height, width, _ = video_frames[0].shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = 30.0
                video_writer_fallback = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
                for frame in video_frames:
                    video_writer_fallback.write(frame)
                video_writer_fallback.release()
                print(f"[INFO]: Saved episode video (fallback): {video_filename}")
            except Exception as e:
                print(f"[WARNING]: Failed to save video for episode {episode_id}: {e}")
        
        # Finish episode
        data_collector.finish_episode(success=success)
        print(f"[INFO]: Episode {episode_id + 1} completed. Success: {success}, Final distance: {final_distance:.4f}")
        
        # Save periodic checkpoints
        saved_file = data_collector.save_periodic("robot_episodes_checkpoint", max_episodes=50)
        if saved_file:
            print(f"[INFO]: Saved checkpoint with {len(data_collector.data['episodes'])} episodes to {saved_file}")
        
        episode_id += 1
    
    # Save final data
    final_file = data_collector.save_data("robot_episodes_final")
    
    # Get comprehensive statistics
    stats = data_collector.get_total_stats()
    
    print(f"\n[INFO]: Data collection complete!")
    print(f"[INFO]: Total episodes processed: {stats['total_episodes']}")
    print(f"[INFO]: Successful episodes: {stats['successful_episodes']}")
    print(f"[INFO]: Success rate: {stats['success_rate']:.1f}%")
    print(f"[INFO]: Final data saved to: {final_file}")
    
    if len(data_collector.data['episodes']) > 0:
        print(f"[INFO]: Final batch contains: {len(data_collector.data['episodes'])} episodes")


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