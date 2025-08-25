import os
import pickle
from datetime import datetime

# Isaac Lab imports for type hints
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import Camera

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