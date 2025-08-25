import argparse
import math
import numpy as np
import torch
import random
import os
import pickle
from datetime import datetime

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

class DataCollector:
    """Collects and saves robot simulation data."""
    
    def __init__(self):
        self.data = {
            'robot_data': [],
            'sphere_data': [],
            'camera_data': [],
            'metadata': {
                'start_time': datetime.now().isoformat(),
                'data_capture_interval': None,
                'sim_dt': None
            }
        }
        self.data_dir = "./collected_data"
        os.makedirs(self.data_dir, exist_ok=True)
    
    def set_metadata(self, data_capture_interval: int, sim_dt: float):
        """Set simulation metadata."""
        self.data['metadata']['data_capture_interval'] = data_capture_interval
        self.data['metadata']['sim_dt'] = sim_dt
    
    def collect_robot_data(self, robot: Articulation, sim_time: float, step_count: int):
        """Collect robot state data."""
        robot_state = {
            'timestamp': sim_time,
            'step_count': step_count,
            'joint_positions': robot.data.joint_pos[0].cpu().numpy().copy(),
            'joint_velocities': robot.data.joint_vel[0].cpu().numpy().copy(),
            'end_effector_position': robot.data.root_pos_w[0].cpu().numpy().copy(),
            'end_effector_orientation': robot.data.root_quat_w[0].cpu().numpy().copy(),
            # Additional robot data you might want:
            'joint_names': ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 
                          'panda_joint5', 'panda_joint6', 'panda_joint7', 'panda_finger_joint1', 'panda_finger_joint2']
        }
        self.data['robot_data'].append(robot_state)
        return robot_state
    
    def collect_sphere_data(self, sphere: RigidObject, sim_time: float, step_count: int):
        """Collect sphere obstacle data."""
        sphere_state = {
            'timestamp': sim_time,
            'step_count': step_count,
            'position': sphere.data.root_pos_w[0].cpu().numpy().copy(),
            'orientation': sphere.data.root_quat_w[0].cpu().numpy().copy(),
        }
        self.data['sphere_data'].append(sphere_state)
        return sphere_state
    
    def collect_camera_data(self, camera: Camera, camera_name: str, sim_time: float, step_count: int):
        """Collect camera data metadata (images saved separately)."""
        camera_state = {
            'timestamp': sim_time,
            'step_count': step_count,
            'camera_name': camera_name,
        }
        
        if camera.data.output is not None:
            if "rgb" in camera.data.output:
                rgb_data = camera.data.output["rgb"][0].cpu().numpy()
                camera_state['rgb_shape'] = rgb_data.shape
                camera_state['rgb_range'] = [float(rgb_data.min()), float(rgb_data.max())]
            
            if "distance_to_image_plane" in camera.data.output:
                depth_data = camera.data.output["distance_to_image_plane"][0].cpu().numpy()
                camera_state['depth_shape'] = depth_data.shape
                camera_state['depth_range'] = [float(depth_data.min()), float(depth_data.max())]
        
        self.data['camera_data'].append(camera_state)
        return camera_state
    
    def save_data(self, filename_prefix: str = "robot_data"):
        """Save collected data to pickle file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.pkl"
        filepath = os.path.join(self.data_dir, filename)
        
        # Add end metadata
        self.data['metadata']['end_time'] = datetime.now().isoformat()
        self.data['metadata']['total_samples'] = len(self.data['robot_data'])
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.data, f)
        
        print(f"[INFO]: Saved {len(self.data['robot_data'])} data samples to {filepath}")
        return filepath
    
    def save_periodic(self, filename_prefix: str = "robot_data_checkpoint", max_samples: int = 1000):
        """Save data periodically and reset buffer to prevent memory issues."""
        if len(self.data['robot_data']) >= max_samples:
            filepath = self.save_data(filename_prefix)
            # Reset data but keep metadata
            metadata_backup = self.data['metadata'].copy()
            self.data = {
                'robot_data': [],
                'sphere_data': [],
                'camera_data': [],
                'metadata': metadata_backup
            }
            return filepath
        return None

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
    
    # Initialize data collector for pickle files
    data_collector = DataCollector()
    data_collector.set_metadata(data_capture_interval, sim_dt)
    
    print("[INFO]: Starting sinusoidal motion simulation...")
    print(f"[INFO]: Data will be saved to pickle files in: {data_collector.data_dir}")
    
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
            # Print robot data and collect for pickle files
            print_robot_data(robot, sim_time, data_collector, count)
            
            # Print sphere data and collect for pickle files
            print_sphere_data(sphere, sim_time, data_collector, count)
            
            # Save camera data and collect metadata for pickle files
            save_camera_data(camera1, "WristCam", sim_time, data_collector, count)
            save_camera_data(camera2, "TableCam", sim_time, data_collector, count)
            
            # Periodically save data to prevent memory issues
            saved_file = data_collector.save_periodic("robot_data_checkpoint", max_samples=200)
            if saved_file:
                print(f"[INFO]: Saved checkpoint data to {saved_file}")
            
            print("-" * 60)
        
        # Update time and counters
        sim_time += sim_dt
        count += 1
        
        # Optional: Stop after certain time for testing
        if sim_time > 5.0:  # Run for 30 seconds
            print("[INFO]: Simulation complete!")
            
            # Save final data to pickle file
            final_file = data_collector.save_data("robot_data_final")
            print(f"[INFO]: Final data saved to {final_file}")
            print(f"[INFO]: Total samples collected: {len(data_collector.data['robot_data'])}")
            
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