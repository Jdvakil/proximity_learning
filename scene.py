import math
import numpy as np
import torch
import random

"""Scene definition for Isaac Lab simulation."""

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Import Franka configuration
from isaaclab_assets import FRANKA_PANDA_CFG

class Scene:
    def __init__(self):
        self.entities = {}

    def generate_random_sphere_position(self, table_center, table_size=(0.8, 0.8), sphere_radius=0.05):
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


    def design_scene(self) -> tuple[dict, dict]:
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
        sphere_position = self.generate_random_sphere_position(table_position)
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