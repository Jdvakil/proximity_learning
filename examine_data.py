#!/usr/bin/env python3
"""
Script to examine the robot data collected in pickle files.
Usage: python examine_data.py <path_to_pickle_file>
"""

import pickle
import argparse
import numpy as np
import os

def examine_pickle_data(pickle_file_path):
    """Load and examine the robot data from a pickle file."""
    
    if not os.path.exists(pickle_file_path):
        print(f"Error: File {pickle_file_path} does not exist!")
        return
    
    print(f"Loading data from: {pickle_file_path}")
    print("=" * 60)
    
    try:
        with open(pickle_file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Print metadata
        print("METADATA:")
        print("-" * 30)
        metadata = data.get('metadata', {})
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        
        # Examine robot data
        robot_data = data.get('robot_data', [])
        print(f"\nROBOT DATA:")
        print("-" * 30)
        print(f"  Number of samples: {len(robot_data)}")
        
        if robot_data:
            first_sample = robot_data[0]
            print(f"  Sample keys: {list(first_sample.keys())}")
            print(f"  Joint positions shape: {first_sample['joint_positions'].shape}")
            print(f"  Joint velocities shape: {first_sample['joint_velocities'].shape}")
            print(f"  End effector position shape: {first_sample['end_effector_position'].shape}")
            print(f"  Time range: {robot_data[0]['timestamp']:.3f}s to {robot_data[-1]['timestamp']:.3f}s")
            
            # Show some statistics
            all_joint_pos = np.array([sample['joint_positions'] for sample in robot_data])
            print(f"  Joint position ranges:")
            for i in range(all_joint_pos.shape[1]):
                joint_name = first_sample['joint_names'][i] if i < len(first_sample['joint_names']) else f"joint_{i}"
                min_val, max_val = all_joint_pos[:, i].min(), all_joint_pos[:, i].max()
                print(f"    {joint_name}: [{min_val:.4f}, {max_val:.4f}]")
        
        # Examine sphere data
        sphere_data = data.get('sphere_data', [])
        print(f"\nSPHERE DATA:")
        print("-" * 30)
        print(f"  Number of samples: {len(sphere_data)}")
        
        if sphere_data:
            print(f"  Sample keys: {list(sphere_data[0].keys())}")
            positions = np.array([sample['position'] for sample in sphere_data])
            print(f"  Position range: X[{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}], "
                  f"Y[{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}], "
                  f"Z[{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]")
        
        # Examine camera data
        camera_data = data.get('camera_data', [])
        print(f"\nCAMERA DATA:")
        print("-" * 30)
        print(f"  Number of samples: {len(camera_data)}")
        
        if camera_data:
            cameras = set(sample['camera_name'] for sample in camera_data)
            print(f"  Cameras: {cameras}")
            for camera_name in cameras:
                camera_samples = [s for s in camera_data if s['camera_name'] == camera_name]
                print(f"  {camera_name}: {len(camera_samples)} samples")
                if camera_samples:
                    sample = camera_samples[0]
                    if 'rgb_shape' in sample:
                        print(f"    RGB shape: {sample['rgb_shape']}")
                    if 'depth_shape' in sample:
                        print(f"    Depth shape: {sample['depth_shape']}")
        
        print("\n" + "=" * 60)
        print("Data examination complete!")
        
    except Exception as e:
        print(f"Error loading pickle file: {e}")

def list_data_files(directory="./collected_data"):
    """List all available pickle files in the data directory."""
    if os.path.exists(directory):
        pickle_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
        if pickle_files:
            print(f"Available data files in {directory}:")
            for i, filename in enumerate(sorted(pickle_files), 1):
                filepath = os.path.join(directory, filename)
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"  {i}. {filename} ({size_mb:.2f} MB)")
        else:
            print(f"No pickle files found in {directory}")
    else:
        print(f"Directory {directory} does not exist")

def main():
    parser = argparse.ArgumentParser(description="Examine robot data from pickle files")
    parser.add_argument("-p", "--pickle_file", nargs='?', help="Path to the pickle file to examine")
    parser.add_argument("--list", action='store_true', help="List available data files")
    args = parser.parse_args()
    
    if args.list:
        list_data_files()
    elif args.pickle_file:
        examine_pickle_data(args.pickle_file)
    else:
        print("Usage: python examine_data.py <pickle_file_path>")
        print("   or: python examine_data.py --list")
        list_data_files()

if __name__ == "__main__":
    main()
