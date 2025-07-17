#!/usr/bin/env python3
"""
Franka Robot Raycast Visualization with MuJoCo Viewer
Enhanced version with both matplotlib visualization and MuJoCo physics simulation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import math
import threading
import xml.etree.ElementTree as ET
import foxglove
# Try to import MuJoCo
try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
    print("✓ MuJoCo available")
except ImportError:
    MUJOCO_AVAILABLE = False
    print("✗ MuJoCo not available - will run matplotlib only")

class FrankaRaycastWithMuJoCo:
    def __init__(self):
        """Initialize the Franka robot with MuJoCo and matplotlib visualization."""
        print("Initializing Franka Raycast Visualization with MuJoCo...")
        
        # Robot parameters
        self.joint_names = [
            'joint1', 'joint2', 'joint3', 'joint4',
            'joint5', 'joint6', 'joint7'
        ]
        
        # Current joint angles
        self.joint_angles = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.7])
        self.target_angles = self.joint_angles.copy()
        
        # Raycasting parameters
        self.raycast_distance = 2.0
        self.num_rays_per_joint = 64
        
        # Colors for joints
        self.joint_colors = [
            'red', 'green', 'blue', 'yellow', 
            'magenta', 'cyan', 'orange'
        ]
        
        # Obstacles for matplotlib version
        self.obstacles = [
            {'center': np.array([1.0, 0.5, 0.5]), 'radius': 0.2, 'color': 'gray'},
            {'center': np.array([-0.5, 1.0, 0.3]), 'radius': 0.15, 'color': 'brown'},
            {'center': np.array([0.8, -0.8, 0.4]), 'radius': 0.25, 'color': 'purple'},
            {'center': np.array([0.2, 0.8, 0.8]), 'radius': 0.18, 'color': 'olive'},
        ]
        
        # Initialize MuJoCo if available
        self.mujoco_initialized = False
        if MUJOCO_AVAILABLE:
            self.setup_mujoco()
        
        # Initialize matplotlib
        self.setup_matplotlib()
        
        # Threading for MuJoCo viewer
        self.mujoco_thread = None
        self.running = True
        
    def create_mujoco_xml(self):
        """Create a MuJoCo XML model for the Franka robot."""
        xml_content = """
                <mujoco model="franka_raycast_integrated">
            <compiler angle="radian" meshdir="./robohive/robohive/simhive/franka_sim" texturedir="./robohive/robohive/simhive/franka_sim"/>

            <size njmax='1000' nconmax='1000'/>

            <!-- <include file="./robohive/robohive/simhive/scene_sim/topfloor_scene.xml"/> -->
            <include file="./robohive/robohive/simhive/franka_sim/assets/assets.xml"/>
            <!-- <include file="./robohive/robohive/simhive/furniture_sim/simpleTable/simpleTable_asset.xml"/> -->
            <include file="./robohive/robohive/simhive/franka_sim/assets/actuator0.xml"/>
            <include file="./robohive/robohive/simhive/franka_sim/assets/gripper_assets.xml"/>
            <include file="./robohive/robohive/simhive/franka_sim/assets/gripper_actuator0.xml"/>
            
            <option timestep="0.01" gravity="0 0 -9.81"/>
            
            <asset>
                <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" 
                        rgb2=".2 .3 .4" width="300" height="300"/>
                <material name="grid" texture="grid" texrepeat="8 8" reflectance=".2"/>
                <material name="robot" rgba="0.7 0.7 0.7 1"/>
                <material name="obstacle_gray" rgba="0.5 0.5 0.5 0.7"/>
                <material name="obstacle_brown" rgba="0.6 0.3 0.1 0.7"/>
                <material name="obstacle_purple" rgba="0.5 0.0 0.5 0.7"/>
                <material name="obstacle_olive" rgba="0.5 0.5 0.0 0.7"/>
                <material name="franka_white" rgba="1 1 1 1"/>
                <material name="franka_black" rgba="0.1 0.1 0.1 1"/>
            </asset>
            
            <worldbody>
                <light pos="0 0 3" dir="0 0 -1"/>
                <geom name="floor" size="3 3 0.1" type="plane" material="grid"/>
                
                <!-- Franka Robot (based on RoboPen structure) -->
                <body pos='0 0 0' euler='0 0 1.57'>
                    <!-- <geom type='cylinder' size='.120 .4' pos='-.04 0 -.4'/> -->
                    <include file="./robohive/robohive/simhive/franka_sim/assets/chain0.xml"/>
                </body>
                
                <!-- Camera positions (from RoboPen) -->
                <camera name='left_cam' pos='-0.5 1.2 1.8' quat='-0.32 -0.22 0.49 0.78'/>
                <camera name='right_cam' pos='-0.5 -1.2 1.8' quat='0.76 0.5 -0.21 -0.35'/>
                <camera name='top_cam' pos='0.5 0 2.2' euler='0 0 -1.57'/>
                
                <!-- Workspace site -->
                <site name='workspace' type='box' size='.375 .6 .25' pos='0.475 0 1.0' group='3' rgba='0 0 1 0.3'/>
                
                <!-- End effector target -->
                <site name='ee_target' type='box' size='.03 .07 .04' pos='0.4 0 1' group='1' rgba='0 1 .4 0.5' euler="0 3.14 3.14"/>
                
                <!-- Obstacles -->
                <body name="obstacle1" pos="1.0 0.5 0.5">
                    <geom name="obs1" type="sphere" size="0.2" material="obstacle_gray"/>
                </body>
                <body name="obstacle2" pos="-0.5 1.0 0.3">
                    <geom name="obs2" type="sphere" size="0.15" material="obstacle_brown"/>
                </body>
                <body name="obstacle3" pos="0.8 -0.8 0.4">
                    <geom name="obs3" type="sphere" size="0.25" material="obstacle_purple"/>
                </body>
                <body name="obstacle4" pos="0.2 0.8 0.8">
                    <geom name="obs4" type="sphere" size="0.18" material="obstacle_olive"/>
                </body>
            </worldbody>
            
        </mujoco>
        """
        return xml_content
    
    def setup_mujoco(self):
        """Setup MuJoCo simulation."""
        try:
            # Create XML model
            xml_content = self.create_mujoco_xml()
            
            # Load model
            self.model = mujoco.MjModel.from_xml_string(xml_content)
            self.data = mujoco.MjData(self.model)
            
            # Set initial joint positions
            for i, angle in enumerate(self.joint_angles):
                if i < len(self.data.qpos):
                    self.data.qpos[i] = angle
            
            # Forward kinematics
            mujoco.mj_forward(self.model, self.data)
            
            print("✓ MuJoCo model loaded successfully")
            self.mujoco_initialized = True
            
        except Exception as e:
            print(f"✗ MuJoCo setup failed: {e}")
            self.mujoco_initialized = False
    
    def setup_matplotlib(self):
        """Setup matplotlib visualization."""
        self.fig = plt.figure(figsize=(16, 12))
        self.ax_3d = self.fig.add_subplot(221, projection='3d')
        self.ax_distances = self.fig.add_subplot(222)
        self.ax_top = self.fig.add_subplot(223)
        self.ax_side = self.fig.add_subplot(224)
        
        self.setup_plots()
    
    def setup_plots(self):
        """Setup all matplotlib plots."""
        # 3D plot
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        self.ax_3d.set_title('3D Robot with Raycast Visualization')
        self.ax_3d.set_xlim([-1.5, 1.5])
        self.ax_3d.set_ylim([-1.5, 1.5])
        self.ax_3d.set_zlim([0, 1.5])
        
        # Distance plot
        self.ax_distances.set_xlabel('Joint Number')
        self.ax_distances.set_ylabel('Distance to Nearest Obstacle (m)')
        self.ax_distances.set_title('Raycast Distances by Joint')
        self.ax_distances.set_ylim([0, self.raycast_distance])
        
        # Top view
        self.ax_top.set_xlabel('X (m)')
        self.ax_top.set_ylabel('Y (m)')
        self.ax_top.set_title('Top View - XY Plane')
        self.ax_top.set_xlim([-1.5, 1.5])
        self.ax_top.set_ylim([-1.5, 1.5])
        self.ax_top.set_aspect('equal')
        
        # Side view
        self.ax_side.set_xlabel('X (m)')
        self.ax_side.set_ylabel('Z (m)')
        self.ax_side.set_title('Side View - XZ Plane')
        self.ax_side.set_xlim([-1.5, 1.5])
        self.ax_side.set_ylim([0, 1.5])
        self.ax_side.set_aspect('equal')
    
    def get_mujoco_joint_positions(self):
        """Get joint positions from MuJoCo simulation."""
        if not self.mujoco_initialized:
            return self.forward_kinematics_simple(self.joint_angles)
        
        positions = []
        
        # Get positions of all bodies
        body_names = ['base', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'end_effector']
        
        for body_name in body_names:
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                if body_id >= 0:
                    pos = self.data.xpos[body_id].copy()
                    positions.append(pos)
            except:
                # If body not found, use simple kinematics
                pass
        
        if len(positions) < 8:
            return self.forward_kinematics_simple(self.joint_angles)
        
        return np.array(positions)
    
    def forward_kinematics_simple(self, joint_angles):
        """Simple forward kinematics for fallback."""
        positions = []
        current_pos = np.array([0.0, 0.0, 0.1])
        current_rot = np.eye(3)
        
        link_lengths = [0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.107]
        
        positions.append(current_pos.copy())
        
        for i, (angle, length) in enumerate(zip(joint_angles, link_lengths)):
            if i == 0:  # Base rotation
                current_rot = np.array([
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1]
                ])
                current_pos += np.array([0, 0, length])
            elif i == 1:  # Shoulder pitch
                rot_y = np.array([
                    [np.cos(angle), 0, np.sin(angle)],
                    [0, 1, 0],
                    [-np.sin(angle), 0, np.cos(angle)]
                ])
                current_rot = current_rot @ rot_y
                current_pos += current_rot @ np.array([0, 0, length])
            elif i == 2:  # Elbow
                rot_y = np.array([
                    [np.cos(angle), 0, np.sin(angle)],
                    [0, 1, 0],
                    [-np.sin(angle), 0, np.cos(angle)]
                ])
                current_rot = current_rot @ rot_y
                current_pos += current_rot @ np.array([length, 0, 0])
            else:  # Other joints
                if i % 2 == 0:
                    rot_y = np.array([
                        [np.cos(angle), 0, np.sin(angle)],
                        [0, 1, 0],
                        [-np.sin(angle), 0, np.cos(angle)]
                    ])
                    current_rot = current_rot @ rot_y
                else:
                    rot_z = np.array([
                        [np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle), np.cos(angle), 0],
                        [0, 0, 1]
                    ])
                    current_rot = current_rot @ rot_z
                current_pos += current_rot @ np.array([length, 0, 0])
            
            positions.append(current_pos.copy())
        
        return np.array(positions)
    
    def get_joint_directions(self, joint_pos):
        """Get multiple raycast directions from a joint position."""
        directions = []
        
        for i in range(self.num_rays_per_joint):
            angle = 2 * np.pi * i / self.num_rays_per_joint
            
            x_dir = np.cos(angle)
            y_dir = np.sin(angle)
            z_dir = 0.1 * np.sin(2 * angle)
            
            direction = np.array([x_dir, y_dir, z_dir])
            direction = direction / np.linalg.norm(direction)
            directions.append(direction)
            
        return directions
    
    def perform_raycast_mujoco(self, start_pos, direction, max_distance):
        """Perform raycast using MuJoCo if available."""
        if not self.mujoco_initialized:
            return self.perform_raycast_simple(start_pos, direction, max_distance)
        
        end_pos = start_pos + direction * max_distance
        
        try:
            # MuJoCo raycast
            geom_id = mujoco.mj_ray(self.model, self.data, start_pos, direction, 
                                   None, 1, -1, 0)
            
            if geom_id >= 0:
                # Get intersection point
                intersection = self.data.contact[0].pos if self.data.ncon > 0 else end_pos
                distance = np.linalg.norm(intersection - start_pos)
                return intersection, distance, geom_id
            else:
                return end_pos, max_distance, -1
                
        except Exception as e:
            # Fallback to simple raycast
            return self.perform_raycast_simple(start_pos, direction, max_distance)
    
    def perform_raycast_simple(self, start_pos, direction, max_distance):
        """Simple raycast for fallback."""
        min_distance = max_distance
        hit_obstacle = None
        
        for obstacle in self.obstacles:
            to_obstacle = obstacle['center'] - start_pos
            proj_length = np.dot(to_obstacle, direction)
            
            if proj_length > 0:
                closest_point = start_pos + direction * proj_length
                distance_to_ray = np.linalg.norm(obstacle['center'] - closest_point)
                
                if distance_to_ray <= obstacle['radius']:
                    intersection_dist = proj_length - np.sqrt(
                        obstacle['radius']**2 - distance_to_ray**2
                    )
                    
                    if intersection_dist > 0 and intersection_dist < min_distance:
                        min_distance = intersection_dist
                        hit_obstacle = obstacle
        
        hit_pos = start_pos + direction * min_distance
        return hit_pos, min_distance, hit_obstacle
    
    def calculate_raycast_results(self, joint_positions):
        """Calculate raycast results for all joints."""
        results = []
        
        for i, joint_pos in enumerate(joint_positions[1:]):
            directions = self.get_joint_directions(joint_pos)
            
            min_distance = float('inf')
            closest_hit_pos = None
            hit_obstacle = None
            
            for direction in directions:
                if self.mujoco_initialized:
                    hit_pos, distance, geom_id = self.perform_raycast_mujoco(
                        joint_pos, direction, self.raycast_distance
                    )
                else:
                    hit_pos, distance, obstacle = self.perform_raycast_simple(
                        joint_pos, direction, self.raycast_distance
                    )
                
                if distance < min_distance:
                    min_distance = distance
                    closest_hit_pos = hit_pos
                    hit_obstacle = obstacle if not self.mujoco_initialized else geom_id
            
            results.append({
                'joint_id': i,
                'joint_pos': joint_pos,
                'hit_pos': closest_hit_pos,
                'distance': min_distance if min_distance < self.raycast_distance else self.raycast_distance,
                'hit_obstacle': hit_obstacle,
                'color': self.joint_colors[i % len(self.joint_colors)]
            })
        
        return results
    
    def animate_robot(self, t):
        """Animate the robot with sinusoidal motion."""
        for i in range(len(self.joint_angles)):
            amplitude = 0.3 if i < 6 else 0.2
            frequency = 0.5 + i * 0.1
            offset = [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.7][i]
            
            self.target_angles[i] = offset + amplitude * np.sin(frequency * t)
        
        # Update MuJoCo simulation
        if self.mujoco_initialized:
            for i, angle in enumerate(self.target_angles):
                if i < len(self.data.qpos):
                    self.data.qpos[i] = angle
            
            mujoco.mj_forward(self.model, self.data)
            
            # Update joint angles from MuJoCo
            self.joint_angles = self.data.qpos[:7].copy()
        else:
            self.joint_angles = self.target_angles.copy()
    
    def run_mujoco_viewer(self):
        """Run MuJoCo viewer in separate thread."""
        if not self.mujoco_initialized:
            return
        
        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                viewer.cam.azimuth = 45
                viewer.cam.elevation = -30
                viewer.cam.distance = 3
                
                start_time = time.time()
                while self.running and viewer.is_running():
                    t = time.time() - start_time
                    
                    # Update robot animation
                    self.animate_robot(t)
                    
                    # Step simulation
                    mujoco.mj_step(self.model, self.data)
                    
                    # Sync viewer
                    viewer.sync()
                    
                    time.sleep(0.01)  # ~100 FPS
                    
        except Exception as e:
            print(f"MuJoCo viewer error: {e}")
    
    def update_plots(self, frame):
        """Update matplotlib plots."""
        # Clear plots
        self.ax_3d.clear()
        self.ax_distances.clear()
        self.ax_top.clear()
        self.ax_side.clear()
        
        # Re-setup plots
        self.setup_plots()
        
        # Get current robot state
        if self.mujoco_initialized:
            joint_positions = self.get_mujoco_joint_positions()
        else:
            t = frame * 0.1
            self.animate_robot(t)
            joint_positions = self.forward_kinematics_simple(self.joint_angles)
        
        # Calculate raycast results
        raycast_results = self.calculate_raycast_results(joint_positions)
        
        # Plot obstacles
        for obstacle in self.obstacles:
            center = obstacle['center']
            radius = obstacle['radius']
            color = obstacle['color']
            
            # 3D sphere
            u = np.linspace(0, 2 * np.pi, 15)
            v = np.linspace(0, np.pi, 15)
            x_sphere = center[0] + radius * np.outer(np.cos(u), np.sin(v))
            y_sphere = center[1] + radius * np.outer(np.sin(u), np.sin(v))
            z_sphere = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
            self.ax_3d.plot_surface(x_sphere, y_sphere, z_sphere, color=color, alpha=0.3)
            
            # 2D projections
            circle_top = plt.Circle((center[0], center[1]), radius, color=color, alpha=0.5)
            self.ax_top.add_patch(circle_top)
            
            circle_side = plt.Circle((center[0], center[2]), radius, color=color, alpha=0.5)
            self.ax_side.add_patch(circle_side)
        
        # Plot robot arm
        x_coords = joint_positions[:, 0]
        y_coords = joint_positions[:, 1]
        z_coords = joint_positions[:, 2]
        
        self.ax_3d.plot(x_coords, y_coords, z_coords, 'ko-', linewidth=3, markersize=8)
        self.ax_top.plot(x_coords, y_coords, 'ko-', linewidth=2, markersize=6)
        self.ax_side.plot(x_coords, z_coords, 'ko-', linewidth=2, markersize=6)
        
        # Plot raycast results
        joint_numbers = []
        distances = []
        colors = []
        
        for result in raycast_results:
            joint_pos = result['joint_pos']
            hit_pos = result['hit_pos']
            distance = result['distance']
            color = result['color']
            
            if hit_pos is not None:
                # 3D raycast line
                self.ax_3d.plot([joint_pos[0], hit_pos[0]], 
                               [joint_pos[1], hit_pos[1]], 
                               [joint_pos[2], hit_pos[2]], 
                               color=color, linewidth=2, alpha=0.7)
                
                self.ax_3d.scatter(hit_pos[0], hit_pos[1], hit_pos[2], 
                                 color=color, s=50, alpha=0.8)
                
                # 2D projections
                self.ax_top.plot([joint_pos[0], hit_pos[0]], 
                                [joint_pos[1], hit_pos[1]], 
                                color=color, linewidth=1, alpha=0.7)
                
                self.ax_side.plot([joint_pos[0], hit_pos[0]], 
                                 [joint_pos[2], hit_pos[2]], 
                                 color=color, linewidth=1, alpha=0.7)
            
            joint_numbers.append(result['joint_id'] + 1)
            distances.append(distance)
            colors.append(color)
        
        # Distance bar chart
        if joint_numbers:
            bars = self.ax_distances.bar(joint_numbers, distances, color=colors, alpha=0.7)
            
            for bar, dist in zip(bars, distances):
                height = bar.get_height()
                self.ax_distances.text(bar.get_x() + bar.get_width()/2, height + 0.05,
                                     f'{dist:.2f}m', ha='center', va='bottom', fontsize=8)
        
        # Add status text
        status = "MuJoCo Active" if self.mujoco_initialized else "Matplotlib Only"
        self.ax_3d.text2D(0.02, 0.98, f'Frame: {frame} | {status}', 
                         transform=self.ax_3d.transAxes, fontsize=10, verticalalignment='top')
        
        # Print distances periodically
        if frame % 50 == 0:
            print(f"\nFrame {frame}:")
            for result in raycast_results:
                print(f"  Joint {result['joint_id']+1}: {result['distance']:.3f}m")
    
    def run_visualization(self, duration=60):
        """Run the complete visualization."""
        print("\n=== Starting Franka Raycast Visualization ===")
        
        if self.mujoco_initialized:
            print("✓ MuJoCo viewer will open in separate window")
            print("✓ Matplotlib plots will show raycast analysis")
            
            # Start MuJoCo viewer in separate thread
            self.mujoco_thread = threading.Thread(target=self.run_mujoco_viewer)
            self.mujoco_thread.daemon = True
            self.mujoco_thread.start()
        else:
            print("✓ Running matplotlib visualization only")
            print("  (Install mujoco to see 3D physics simulation)")
        
        # Setup matplotlib animation
        fps = 10
        frames = duration * fps
        
        anim = FuncAnimation(self.fig, self.update_plots, frames=frames, 
                           interval=1000//fps, blit=False, repeat=True)
        
        plt.tight_layout()
        plt.show()
        
        # Stop MuJoCo viewer
        self.running = False
        if self.mujoco_thread:
            self.mujoco_thread.join(timeout=1)
        
        return anim

def main():
    """Main function."""
    print("=== Franka Raycast Visualization with MuJoCo ===")
    print("This demonstrates:")
    print("• MuJoCo physics simulation with 3D robot model")
    print("• Raycast-based obstacle detection")
    print("• Real-time distance measurements")
    print("• Multiple visualization views")
    print()
    
    if not MUJOCO_AVAILABLE:
        print("Note: MuJoCo not available. Install with:")
        print("pip install mujoco")
        print("Running with matplotlib visualization only.")
        print()
    
    try:
        visualizer = FrankaRaycastWithMuJoCo()
        anim = visualizer.run_visualization(duration=60)
        
        print("\n=== Visualization Complete ===")
        if MUJOCO_AVAILABLE:
            print("✓ MuJoCo physics simulation")
        print("✓ Raycast obstacle detection")
        print("✓ Multi-view analysis")
        print("✓ Real-time distance measurements")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()