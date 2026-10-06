import time
import numpy as np
import torch
import gymnasium as gym
from envs.active_slam_env import ActiveSLAMEnv
from stable_baselines3 import PPO

def measure_inference():
    print("Loading environment...")
    env = ActiveSLAMEnv(
        map_resolution=50,
        num_rays=128
    )
    print("Loading PPO model...")
    model = PPO.load("active_slam_ppo.zip")
    
    obs, info = env.reset(seed=42)
    
    # Warmup
    for _ in range(10):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

    # Measure raw PyTorch model prediction time
    print("Measuring raw policy model.predict inference time...")
    inference_times = []
    for _ in range(500):
        t0 = time.perf_counter()
        action, _ = model.predict(obs, deterministic=True)
        t1 = time.perf_counter()
        inference_times.append(t1 - t0)
        
        # Step env to keep obs updated
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
            
    print(f"Policy Inference Latency (predict):")
    print(f"  Mean:   {np.mean(inference_times)*1000:.3f} ms")
    print(f"  Median: {np.median(inference_times)*1000:.3f} ms")
    print(f"  StdDev: {np.std(inference_times)*1000:.3f} ms")
    
    # Measure full env step time
    print("Measuring full env.step time...")
    step_times = []
    obs, info = env.reset()
    for _ in range(500):
        action, _ = model.predict(obs, deterministic=True)
        
        t0 = time.perf_counter()
        obs, reward, terminated, truncated, info = env.step(action)
        t1 = time.perf_counter()
        step_times.append(t1 - t0)
        
        if terminated or truncated:
            obs, info = env.reset()
            
    print(f"Environment Step Latency (env.step):")
    print(f"  Mean:   {np.mean(step_times)*1000:.3f} ms")
    print(f"  Median: {np.median(step_times)*1000:.3f} ms")
    print(f"  StdDev: {np.std(step_times)*1000:.3f} ms")

if __name__ == "__main__":
    measure_inference()
