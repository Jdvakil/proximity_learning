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
            'episodes': [],  # List of episode dictionaries
            'metadata': {
                'start_time': datetime.now().isoformat(),
                'episode_length': None,
                'sim_dt': None,
                'total_episodes': 0,
                'successful_episodes': 0
            }
        }
        self.current_episode = None
        self.total_episodes_processed = 0  # Track total across all saves
        self.total_successful_episodes = 0  # Track successful across all saves
        self.data_dir = "./collected_data"
        os.makedirs(self.data_dir, exist_ok=True)
    
    def start_new_episode(self, episode_id: int, start_joint_pos, goal_joint_pos):
        """Start collecting data for a new episode."""
        self.current_episode = {
            'episode_id': episode_id,
            'start_joint_positions': start_joint_pos.cpu().numpy().copy(),
            'goal_joint_positions': goal_joint_pos.cpu().numpy().copy(),
            'trajectory': [],  # List of robot states during the episode
            'sphere_data': [],  # Sphere positions during the episode
            'camera_data': [],  # Camera data during the episode
            'episode_metadata': {
                'start_time': datetime.now().isoformat(),
                'episode_length': None,
                'success': False
            }
        }
    
    def collect_step_data(self, robot: Articulation, sphere: RigidObject, cameras: dict, sim_time: float, step_in_episode: int):
        """Collect data for one step within the current episode."""
        if self.current_episode is None:
            raise ValueError("No episode started. Call start_new_episode first.")
        
        # Collect robot data
        robot_state = {
            'step': step_in_episode,
            'timestamp': sim_time,
            'joint_positions': robot.data.joint_pos[0].cpu().numpy().copy(),
            'joint_velocities': robot.data.joint_vel[0].cpu().numpy().copy(),
            'end_effector_position': robot.data.root_pos_w[0].cpu().numpy().copy(),
            'end_effector_orientation': robot.data.root_quat_w[0].cpu().numpy().copy(),
        }
        self.current_episode['trajectory'].append(robot_state)
        
        # Collect sphere data
        sphere_state = {
            'step': step_in_episode,
            'timestamp': sim_time,
            'position': sphere.data.root_pos_w[0].cpu().numpy().copy(),
            'orientation': sphere.data.root_quat_w[0].cpu().numpy().copy(),
        }
        self.current_episode['sphere_data'].append(sphere_state)
        
        # Collect camera data metadata
        for camera_name, camera in cameras.items():
            if camera.data.output is not None:
                camera_state = {
                    'step': step_in_episode,
                    'timestamp': sim_time,
                    'camera_name': camera_name,
                }
                
                if "rgb" in camera.data.output:
                    rgb_data = camera.data.output["rgb"][0].cpu().numpy()
                    camera_state['rgb_shape'] = rgb_data.shape
                    camera_state['rgb_range'] = [float(rgb_data.min()), float(rgb_data.max())]
                
                if "distance_to_image_plane" in camera.data.output:
                    depth_data = camera.data.output["distance_to_image_plane"][0].cpu().numpy()
                    camera_state['depth_shape'] = depth_data.shape
                    camera_state['depth_range'] = [float(depth_data.min()), float(depth_data.max())]
                
                self.current_episode['camera_data'].append(camera_state)
    
    def finish_episode(self, success: bool = True):
        """Finish the current episode and add it to the dataset."""
        if self.current_episode is None:
            raise ValueError("No episode to finish.")
        
        self.current_episode['episode_metadata']['end_time'] = datetime.now().isoformat()
        self.current_episode['episode_metadata']['episode_length'] = len(self.current_episode['trajectory'])
        self.current_episode['episode_metadata']['success'] = success
        
        # Add episode to the main data structure
        self.data['episodes'].append(self.current_episode)
        self.data['metadata']['total_episodes'] += 1
        
        # Track totals across all saves
        self.total_episodes_processed += 1
        if success:
            self.total_successful_episodes += 1
            self.data['metadata']['successful_episodes'] += 1
        
        self.current_episode = None
    
    def set_metadata(self, episode_length: int, sim_dt: float):
        """Set simulation metadata."""
        self.data['metadata']['episode_length'] = episode_length
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
    
    def save_data(self, filename_prefix: str = "robot_episodes"):
        """Save collected episode data to pickle file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.pkl"
        filepath = os.path.join(self.data_dir, filename)
        
        # Add end metadata
        self.data['metadata']['end_time'] = datetime.now().isoformat()
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.data, f)
        
        print(f"[INFO]: Saved {len(self.data['episodes'])} episodes to {filepath}")
        return filepath
    
    def save_periodic(self, filename_prefix: str = "robot_episodes_checkpoint", max_episodes: int = 100):
        """Save data periodically and reset buffer to prevent memory issues."""
        if len(self.data['episodes']) >= max_episodes:
            filepath = self.save_data(filename_prefix)
            # Reset data but keep metadata and totals
            metadata_backup = self.data['metadata'].copy()
            metadata_backup['total_episodes'] = 0  # Reset count for this batch
            metadata_backup['successful_episodes'] = 0  # Reset count for this batch
            self.data = {
                'episodes': [],
                'metadata': metadata_backup
            }
            return filepath
        return None
    
    def get_total_stats(self):
        """Get total statistics across all episodes processed."""
        success_rate = 0.0
        if self.total_episodes_processed > 0:
            success_rate = (self.total_successful_episodes / self.total_episodes_processed) * 100
        
        return {
            'total_episodes': self.total_episodes_processed,
            'successful_episodes': self.total_successful_episodes,
            'success_rate': success_rate
        }