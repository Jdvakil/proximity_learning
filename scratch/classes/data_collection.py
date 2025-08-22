#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Modular Isaac Lab Framework for Franka Robot Simulation with Obstacle Avoidance
and Comprehensive Data Collection

This modular framework allows easy configuration of:
- Different robots and controllers
- Various obstacle configurations
- Multiple camera setups
- Flexible data collection strategies
- HDF5 data saving

Usage:
    ./isaaclab.sh -p scripts/demos/modular_franka_sim.py --config_file configs/default.yaml
"""

import argparse
import math
import numpy as np
import torch
import random
import h5py
import cv2
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Modular Franka robot simulation framework.")
parser.add_argument("--config_file", type=str, default=None, help="Path to YAML configuration file")
parser.add_argument("--save_data", action="store_true", help="Enable HDF5 data saving")
parser.add_argument("--data_dir", type=str, default="./data", help="Directory to save data")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Rest of imports after app launch
import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets import FRANKA_PANDA_CFG


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class CameraConfig:
    """Configuration for camera setup."""
    name: str
    prim_path: str
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float, float]
    height: int = 480
    width: int = 640
    update_period: float = 0.1
    data_types: List[str] = field(default_factory=lambda: ["rgb", "distance_to_image_plane"])
    focal_length: float = 24.0
    focus_distance: float = 400.0
    horizontal_aperture: float = 20.955
    clipping_range: Tuple[float, float] = (0.1, 2.0)
    convention: str = "opengl"

@dataclass
class ObstacleConfig:
    """Configuration for obstacles."""
    name: str
    shape: str  # "sphere", "box", "cylinder"
    size: Tuple[float, ...] # radius for sphere, (width, height, depth) for box, (radius, height) for cylinder
    position: Tuple[float, float, float]
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    kinematic: bool = True
    randomize_position: bool = True
    randomize_bounds: Tuple[Tuple[float, float], ...] = field(default_factory=lambda: ((-0.3, 0.3), (-0.3, 0.3), (0.0, 0.0)))

@dataclass
class RobotConfig:
    """Configuration for robot setup."""
    name: str = "franka_panda"
    position: Tuple[float, float, float] = (0.0, 0.0, 1.05)
    controller_type: str = "sinusoidal"  # "sinusoidal", "random", "collision_avoidance"

@dataclass
class SimulationConfig:
    """Main simulation configuration."""
    dt: float = 1/60.0
    max_time: float = 30.0
    reset_interval: int = 1000
    data_capture_interval: int = 10
    device: str = "cuda:0"
    
    # Scene components
    robot: RobotConfig = field(default_factory=RobotConfig)
    cameras: List[CameraConfig] = field(default_factory=list)
    obstacles: List[ObstacleConfig] = field(default_factory=list)
    
    # Data collection
    save_data: bool = True
    data_dir: str = "./data"
    save_images: bool = True
    save_robot_state: bool = True
    save_obstacles_state: bool = True


# =============================================================================
# Base Classes and Interfaces
# =============================================================================

class Controller(ABC):
    """Abstract base class for robot controllers."""
    
    @abstractmethod
    def compute_target_positions(self, robot: Articulation, sim_time: float, **kwargs) -> torch.Tensor:
        """Compute target joint positions for the robot."""
        pass

class ObstacleManager(ABC):
    """Abstract base class for obstacle management."""
    
    @abstractmethod
    def create_obstacles(self, configs: List[ObstacleConfig]) -> Dict[str, RigidObject]:
        """Create obstacles based on configurations."""
        pass
    
    @abstractmethod
    def update_obstacles(self, obstacles: Dict[str, RigidObject], sim_time: float):
        """Update obstacle states."""
        pass

class DataCollector(ABC):
    """Abstract base class for data collection."""
    
    @abstractmethod
    def collect_data(self, entities: Dict, sim_time: float, step: int):
        """Collect simulation data."""
        pass
    
    @abstractmethod
    def save_data(self, filepath: str):
        """Save collected data to file."""
        pass


# =============================================================================
# Controller Implementations
# =============================================================================

class SinusoidalController(Controller):
    """Sinusoidal motion controller."""
    
    def __init__(self, amplitude: float = 0.3, frequency: float = 0.5):
        self.amplitude = amplitude
        self.frequency = frequency
    
    def compute_target_positions(self, robot: Articulation, sim_time: float, **kwargs) -> torch.Tensor:
        """Apply sinusoidal motion to robot joints."""
        current_joint_pos = robot.data.default_joint_pos.clone()
        joint_offsets = torch.zeros_like(current_joint_pos)
        
        # Apply sinusoidal motion to different joints with different phases
        joint_offsets[:, 0] = self.amplitude * math.sin(2 * math.pi * self.frequency * sim_time)
        joint_offsets[:, 1] = self.amplitude * 0.5 * math.sin(2 * math.pi * self.frequency * sim_time + math.pi/4)
        joint_offsets[:, 2] = self.amplitude * 0.7 * math.sin(2 * math.pi * self.frequency * sim_time + math.pi/2)
        joint_offsets[:, 3] = self.amplitude * 0.4 * math.sin(2 * math.pi * self.frequency * sim_time + 3*math.pi/4)
        joint_offsets[:, 4] = self.amplitude * 0.3 * math.sin(2 * math.pi * self.frequency * sim_time + math.pi)
        joint_offsets[:, 5] = self.amplitude * 0.2 * math.sin(2 * math.pi * self.frequency * sim_time + 5*math.pi/4)
        joint_offsets[:, 6] = self.amplitude * 0.1 * math.sin(2 * math.pi * self.frequency * sim_time + 3*math.pi/2)
        
        joint_pos_target = current_joint_pos + joint_offsets
        
        # Clamp to joint limits
        joint_pos_target = joint_pos_target.clamp_(
            robot.data.soft_joint_pos_limits[..., 0], 
            robot.data.soft_joint_pos_limits[..., 1]
        )
        
        return joint_pos_target

class RandomController(Controller):
    """Random motion controller."""
    
    def __init__(self, amplitude: float = 0.2, update_frequency: float = 1.0):
        self.amplitude = amplitude
        self.update_frequency = update_frequency
        self.last_update_time = 0.0
        self.target_offsets = None
    
    def compute_target_positions(self, robot: Articulation, sim_time: float, **kwargs) -> torch.Tensor:
        """Apply random motion to robot joints."""
        current_joint_pos = robot.data.default_joint_pos.clone()
        
        # Update random targets periodically
        if (sim_time - self.last_update_time) > (1.0 / self.update_frequency) or self.target_offsets is None:
            self.target_offsets = torch.randn_like(current_joint_pos) * self.amplitude
            self.last_update_time = sim_time
        
        joint_pos_target = current_joint_pos + self.target_offsets
        
        # Clamp to joint limits
        joint_pos_target = joint_pos_target.clamp_(
            robot.data.soft_joint_pos_limits[..., 0], 
            robot.data.soft_joint_pos_limits[..., 1]
        )
        
        return joint_pos_target


# =============================================================================
# Obstacle Management
# =============================================================================

class StandardObstacleManager(ObstacleManager):
    """Standard obstacle manager implementation."""
    
    def __init__(self, table_position: Tuple[float, float, float] = (0.55, 0.0, 1.05)):
        self.table_position = table_position
    
    def create_obstacles(self, configs: List[ObstacleConfig]) -> Dict[str, RigidObject]:
        """Create obstacles based on configurations."""
        obstacles = {}
        
        for i, config in enumerate(configs):
            obstacles[config.name] = self._create_single_obstacle(config, i)
        
        return obstacles
    
    def _create_single_obstacle(self, config: ObstacleConfig, index: int) -> RigidObject:
        """Create a single obstacle."""
        prim_path = f"/World/Origin1/{config.name}_{index}"
        
        # Create spawn configuration based on shape
        if config.shape == "sphere":
            spawn_cfg = sim_utils.SphereCfg(radius=config.size[0])
        elif config.shape == "box":
            spawn_cfg = sim_utils.CuboidCfg(size=config.size)
        elif config.shape == "cylinder":
            spawn_cfg = sim_utils.CylinderCfg(radius=config.size[0], height=config.size[1])
        else:
            raise ValueError(f"Unknown obstacle shape: {config.shape}")
        
        # Configure physics properties
        if config.kinematic:
            rigid_props = sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                solver_position_iteration_count=0,
                solver_velocity_iteration_count=0,
                max_angular_velocity=0.0,
                max_linear_velocity=0.0,
                max_depenetration_velocity=0.0,
            )
        else:
            rigid_props = sim_utils.RigidBodyPropertiesCfg()
        
        # Set spawn configuration properties
        spawn_cfg.rigid_props = rigid_props
        spawn_cfg.mass_props = sim_utils.MassPropertiesCfg(mass=1.0)
        spawn_cfg.collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
        spawn_cfg.visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=config.color,
            metallic=0.2,
            roughness=0.5,
        )
        
        # Create obstacle configuration
        obstacle_cfg = RigidObjectCfg(
            prim_path=prim_path,
            spawn=spawn_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=config.position,
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )
        
        return RigidObject(cfg=obstacle_cfg)
    
    def update_obstacles(self, obstacles: Dict[str, RigidObject], sim_time: float):
        """Update obstacle states (implement dynamic obstacles if needed)."""
        pass
    
    def randomize_obstacle_position(self, obstacle: RigidObject, config: ObstacleConfig):
        """Randomize obstacle position within bounds."""
        if not config.randomize_position:
            return
        
        # Generate new random position
        new_pos = list(config.position)
        for i, (min_offset, max_offset) in enumerate(config.randomize_bounds):
            if i < len(new_pos):
                new_pos[i] += random.uniform(min_offset, max_offset)
        
        # Update obstacle position
        obstacle_state = obstacle.data.default_root_state.clone()
        obstacle_state[:, :3] = torch.tensor(new_pos, device=obstacle.device).unsqueeze(0)
        obstacle_state[:, 7:] = 0.0  # Zero velocities
        obstacle.write_root_pose_to_sim(obstacle_state[:, :7])
        obstacle.write_root_velocity_to_sim(obstacle_state[:, 7:])
        obstacle.reset()


# =============================================================================
# Data Collection
# =============================================================================

class HDF5DataCollector(DataCollector):
    """HDF5-based data collector."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.data = {
            'timestamps': [],
            'robot_joint_positions': [],
            'robot_joint_velocities': [],
            'robot_end_effector_pos': [],
            'robot_end_effector_quat': [],
            'camera_data': {},
            'obstacle_data': {}
        }
        self.step_count = 0
    
    def collect_data(self, entities: Dict, sim_time: float, step: int):
        """Collect simulation data."""
        robot = entities["robot"]
        
        # Collect robot data
        if self.config.save_robot_state:
            self.data['timestamps'].append(sim_time)
            self.data['robot_joint_positions'].append(robot.data.joint_pos[0].cpu().numpy())
            self.data['robot_joint_velocities'].append(robot.data.joint_vel[0].cpu().numpy())
            self.data['robot_end_effector_pos'].append(robot.data.root_pos_w[0].cpu().numpy())
            self.data['robot_end_effector_quat'].append(robot.data.root_quat_w[0].cpu().numpy())
        
        # Collect camera data
        for camera_name, camera in entities.get("cameras", {}).items():
            if camera_name not in self.data['camera_data']:
                self.data['camera_data'][camera_name] = {
                    'rgb_images': [],
                    'depth_images': [],
                    'timestamps': []
                }
            
            if camera.data.output is not None:
                camera_data = self.data['camera_data'][camera_name]
                camera_data['timestamps'].append(sim_time)
                
                if "rgb" in camera.data.output:
                    rgb_data = camera.data.output["rgb"][0].cpu().numpy()
                    camera_data['rgb_images'].append(rgb_data)
                
                if "distance_to_image_plane" in camera.data.output:
                    depth_data = camera.data.output["distance_to_image_plane"][0].cpu().numpy()
                    camera_data['depth_images'].append(depth_data)
        
        # Collect obstacle data
        if self.config.save_obstacles_state:
            for obstacle_name, obstacle in entities.get("obstacles", {}).items():
                if obstacle_name not in self.data['obstacle_data']:
                    self.data['obstacle_data'][obstacle_name] = {
                        'positions': [],
                        'orientations': [],
                        'timestamps': []
                    }
                
                obstacle_data = self.data['obstacle_data'][obstacle_name]
                obstacle_data['timestamps'].append(sim_time)
                obstacle_data['positions'].append(obstacle.data.root_pos_w[0].cpu().numpy())
                obstacle_data['orientations'].append(obstacle.data.root_quat_w[0].cpu().numpy())
        
        self.step_count += 1
    
    def save_data(self, filepath: str):
        """Save collected data to HDF5 file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(filepath, 'w') as f:
            # Save metadata
            f.attrs['total_steps'] = self.step_count
            f.attrs['data_capture_interval'] = self.config.data_capture_interval
            f.attrs['simulation_dt'] = self.config.dt
            
            # Save robot data
            if self.data['timestamps']:
                robot_group = f.create_group('robot')
                robot_group.create_dataset('timestamps', data=np.array(self.data['timestamps']))
                robot_group.create_dataset('joint_positions', data=np.array(self.data['robot_joint_positions']))
                robot_group.create_dataset('joint_velocities', data=np.array(self.data['robot_joint_velocities']))
                robot_group.create_dataset('end_effector_positions', data=np.array(self.data['robot_end_effector_pos']))
                robot_group.create_dataset('end_effector_orientations', data=np.array(self.data['robot_end_effector_quat']))
            
            # Save camera data
            if self.data['camera_data']:
                cameras_group = f.create_group('cameras')
                for camera_name, camera_data in self.data['camera_data'].items():
                    cam_group = cameras_group.create_group(camera_name)
                    cam_group.create_dataset('timestamps', data=np.array(camera_data['timestamps']))
                    
                    if camera_data['rgb_images']:
                        cam_group.create_dataset('rgb_images', data=np.array(camera_data['rgb_images']))
                    if camera_data['depth_images']:
                        cam_group.create_dataset('depth_images', data=np.array(camera_data['depth_images']))
            
            # Save obstacle data
            if self.data['obstacle_data']:
                obstacles_group = f.create_group('obstacles')
                for obstacle_name, obstacle_data in self.data['obstacle_data'].items():
                    obs_group = obstacles_group.create_group(obstacle_name)
                    obs_group.create_dataset('timestamps', data=np.array(obstacle_data['timestamps']))
                    obs_group.create_dataset('positions', data=np.array(obstacle_data['positions']))
                    obs_group.create_dataset('orientations', data=np.array(obstacle_data['orientations']))
        
        print(f"[INFO]: Data saved to {filepath}")


# =============================================================================
# Scene Builder
# =============================================================================

class SceneBuilder:
    """Builds simulation scenes based on configuration."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
    
    def build_scene(self) -> Dict[str, Any]:
        """Build the complete simulation scene."""
        # Setup basic scene elements
        self._setup_basic_scene()
        
        # Create robot
        robot = self._create_robot()
        
        # Create cameras
        cameras = self._create_cameras()
        
        # Create obstacles
        obstacle_manager = StandardObstacleManager()
        obstacles = obstacle_manager.create_obstacles(self.config.obstacles)
        
        # Create controller
        controller = self._create_controller()
        
        # Create data collector
        data_collector = None
        if self.config.save_data:
            data_collector = HDF5DataCollector(self.config)
        
        return {
            "robot": robot,
            "cameras": cameras,
            "obstacles": obstacles,
            "controller": controller,
            "obstacle_manager": obstacle_manager,
            "data_collector": data_collector
        }
    
    def _setup_basic_scene(self):
        """Setup ground plane, lights, and table."""
        # Ground plane
        cfg = sim_utils.GroundPlaneCfg()
        cfg.func("/World/defaultGroundPlane", cfg)
        
        # Lights
        cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        cfg.func("/World/Light", cfg)
        
        # Create origin
        prim_utils.create_prim("/World/Origin1", "Xform", translation=[0.0, 0.0, 0.0])
        
        # Table
        table_position = (0.55, 0.0, 1.05)
        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        cfg.func("/World/Origin1/Table", cfg, translation=table_position)
    
    def _create_robot(self) -> Articulation:
        """Create robot based on configuration."""
        robot_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/Origin1/Robot")
        robot_cfg.init_state.pos = self.config.robot.position
        return Articulation(cfg=robot_cfg)
    
    def _create_cameras(self) -> Dict[str, Camera]:
        """Create cameras based on configuration."""
        cameras = {}
        
        for cam_config in self.config.cameras:
            camera_cfg = CameraCfg(
                prim_path=cam_config.prim_path,
                update_period=cam_config.update_period,
                height=cam_config.height,
                width=cam_config.width,
                data_types=cam_config.data_types,
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=cam_config.focal_length,
                    focus_distance=cam_config.focus_distance,
                    horizontal_aperture=cam_config.horizontal_aperture,
                    clipping_range=cam_config.clipping_range
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=cam_config.position,
                    rot=cam_config.rotation,
                    convention=cam_config.convention
                ),
            )
            cameras[cam_config.name] = Camera(cfg=camera_cfg)
        
        return cameras
    
    def _create_controller(self) -> Controller:
        """Create controller based on configuration."""
        if self.config.robot.controller_type == "sinusoidal":
            return SinusoidalController()
        elif self.config.robot.controller_type == "random":
            return RandomController()
        else:
            raise ValueError(f"Unknown controller type: {self.config.robot.controller_type}")


# =============================================================================
# Configuration Loading
# =============================================================================

def load_config(config_file: Optional[str] = None) -> SimulationConfig:
    """Load configuration from file or create default."""
    if config_file and Path(config_file).exists():
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        # Convert dict to config (simplified implementation)
        return SimulationConfig(**config_dict)
    else:
        # Return default configuration
        return SimulationConfig(
            save_data=args_cli.save_data,
            data_dir=args_cli.data_dir,
            cameras=[
                CameraConfig(
                    name="wrist_cam",
                    prim_path="/World/Origin1/Robot/panda_hand/wrist_cam",
                    position=(0.13, 0.0, -0.15),
                    rotation=(-0.70614, 0.03701, 0.03701, -0.70614),
                    convention="ros"
                ),
                CameraConfig(
                    name="table_cam",
                    prim_path="/World/table_cam",
                    position=(2.0, 0.0, 1.6),
                    rotation=(0.56099, 0.43046, 0.43046, 0.56099),
                    convention="opengl"
                )
            ],
            obstacles=[
                ObstacleConfig(
                    name="sphere",
                    shape="sphere",
                    size=(0.05,),
                    position=(0.55, 0.0, 1.12),
                    randomize_position=True,
                    randomize_bounds=((-0.3, 0.3), (-0.3, 0.3), (0.0, 0.0))
                )
            ]
        )


# =============================================================================
# Main Simulation Loop
# =============================================================================

def run_simulation(sim: sim_utils.SimulationContext, entities: Dict[str, Any], config: SimulationConfig):
    """Run the main simulation loop."""
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    
    # Extract entities
    robot = entities["robot"]
    cameras = entities["cameras"]
    obstacles = entities["obstacles"]
    controller = entities["controller"]
    obstacle_manager = entities["obstacle_manager"]
    data_collector = entities["data_collector"]
    
    print("[INFO]: Starting modular simulation...")
    
    while simulation_app.is_running():
        # Reset periodically
        if count % config.reset_interval == 0 and count > 0:
            print(f"[INFO]: Resetting at step {count}...")
            
            # Reset robot
            root_state = robot.data.default_root_state.clone()
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            
            # Randomize obstacles
            for obstacle_name, obstacle in obstacles.items():
                obstacle_config = config.obstacles[0]  # Simplified for demo
                obstacle_manager.randomize_obstacle_position(obstacle, obstacle_config)
            
            sim_time = 0.0
        
        # Control robot
        joint_pos_target = controller.compute_target_positions(robot, sim_time)
        robot.set_joint_position_target(joint_pos_target)
        robot.write_data_to_sim()
        
        # Step simulation
        sim.step()
        
        # Update entities
        robot.update(sim_dt)
        for camera in cameras.values():
            camera.update(sim_dt)
        for obstacle in obstacles.values():
            obstacle.update(sim_dt)
        
        # Collect data
        if data_collector and (count % config.data_capture_interval == 0):
            data_collector.collect_data({
                "robot": robot,
                "cameras": cameras,
                "obstacles": obstacles
            }, sim_time, count)
        
        # Print progress
        if count % 100 == 0:
            print(f"[INFO]: Step {count}, Time: {sim_time:.2f}s")
        
        # Update counters
        sim_time += sim_dt
        count += 1
        
        # Check termination
        if sim_time > config.max_time:
            print("[INFO]: Simulation complete!")
            break
    
    # Save data
    if data_collector:
        timestamp = int(time.time())
        filepath = f"{config.data_dir}/simulation_data_{timestamp}.h5"
        data_collector.save_data(filepath)


# =============================================================================
# Main Function
# =============================================================================

def main():
    """Main function."""
    # Load configuration
    config = load_config(args_cli.config_file)
    config.device = args_cli.device
    if args_cli.save_data:
        config.save_data = True
    if args_cli.data_dir:
        config.data_dir = args_cli.data_dir
    
    # Initialize simulation
    sim_cfg = sim_utils.SimulationCfg(device=config.device, dt=config.dt)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # Set camera view
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    
    # Build scene
    scene_builder = SceneBuilder(config)
    entities = scene_builder.build_scene()
    
    # Reset simulation
    sim.reset()
    
    print("[INFO]: Modular simulation setup complete")
    print(f"[INFO]: Controller: {config.robot.controller_type}")
    print(f"[INFO]: Cameras: {len(config.cameras)}")
    print(f"[INFO]: Obstacles: {len(config.obstacles)}")
    print(f"[INFO]: Data saving: {config.save_data}")
    
    # Run simulation
    run_simulation(sim, entities, config)


if __name__ == "__main__":
    main()
    simulation_app.close()