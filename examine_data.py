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
    """Load and examine the robot episode data from a pickle file."""
    
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
        
        # Examine episodes
        episodes = data.get('episodes', [])
        print(f"\nEPISODES:")
        print("-" * 30)
        print(f"  Total episodes: {len(episodes)}")
        
        if episodes:
            # Show episode structure
            first_episode = episodes[0]
            print(f"  Episode keys: {list(first_episode.keys())}")
            print(f"  Trajectory length: {len(first_episode['trajectory'])}")
            print(f"  Start joint positions shape: {first_episode['start_joint_positions'].shape}")
            print(f"  Goal joint positions shape: {first_episode['goal_joint_positions'].shape}")
            
            # Show success statistics
            successful_episodes = sum(1 for ep in episodes if ep['episode_metadata']['success'])
            success_rate = successful_episodes / len(episodes) * 100
            print(f"  Success rate: {success_rate:.1f}% ({successful_episodes}/{len(episodes)})")
            
            # Show a sample episode
            print(f"\nSAMPLE EPISODE (Episode 0):")
            print("-" * 30)
            sample_ep = episodes[0]
            print(f"  Episode ID: {sample_ep['episode_id']}")
            print(f"  Start joints: {sample_ep['start_joint_positions'][:7]}")  # First 7 joints
            print(f"  Goal joints: {sample_ep['goal_joint_positions'][:7]}")   # First 7 joints
            print(f"  Success: {sample_ep['episode_metadata']['success']}")
            print(f"  Trajectory steps: {len(sample_ep['trajectory'])}")
            
            # Show trajectory statistics
            if sample_ep['trajectory']:
                start_pos = sample_ep['trajectory'][0]['joint_positions'][:7]
                end_pos = sample_ep['trajectory'][-1]['joint_positions'][:7]
                goal_pos = sample_ep['goal_joint_positions'][:7]
                
                final_distance = np.linalg.norm(end_pos - goal_pos)
                print(f"  Final distance to goal: {final_distance:.4f}")
                
                print(f"  Trajectory joint ranges:")
                all_positions = np.array([step['joint_positions'][:7] for step in sample_ep['trajectory']])
                for i in range(7):
                    joint_name = f"joint_{i+1}"
                    min_val, max_val = all_positions[:, i].min(), all_positions[:, i].max()
                    print(f"    {joint_name}: [{min_val:.4f}, {max_val:.4f}]")
        
        # Examine sphere data (if present)
        if episodes and episodes[0]['sphere_data']:
            print(f"\nSPHERE DATA:")
            print("-" * 30)
            sphere_samples = episodes[0]['sphere_data']
            print(f"  Sphere samples per episode: {len(sphere_samples)}")
            if sphere_samples:
                first_sphere = sphere_samples[0]
                print(f"  Sphere position: {first_sphere['position']}")
        
        # Examine camera data (if present)
        if episodes and episodes[0]['camera_data']:
            print(f"\nCAMERA DATA:")
            print("-" * 30)
            camera_samples = episodes[0]['camera_data']
            print(f"  Camera samples per episode: {len(camera_samples)}")
            if camera_samples:
                cameras = set(sample['camera_name'] for sample in camera_samples)
                print(f"  Cameras: {cameras}")
                for camera_name in cameras:
                    camera_data = [s for s in camera_samples if s['camera_name'] == camera_name]
                    print(f"  {camera_name}: {len(camera_data)} samples per episode")
        
        print("\n" + "=" * 60)
        print("Episode data examination complete!")
        
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        import traceback
        traceback.print_exc()

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
