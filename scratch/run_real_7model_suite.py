"""
Run full simulation evaluation for all 7 models on Intel Lab and MIT Stata.
Extracts EXACT real trajectories from the environment and plots real multi-seed coverage curves.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from stable_baselines3 import PPO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.active_slam_env import ActiveSLAMEnv
from envs.adaptive_env import AdaptiveActiveSLAMEnv
from scratch.run_multi_model_benchmark import get_intel_lab_walls, run_yamauchi, run_random_walk, run_rrt_exploration, run_stachniss_entropy
from scratch.run_mit_benchmark import get_mit_stata_walls
from scratch.run_vlm_llm_benchmark import run_shanghai_drl_agent, run_vlm_agent, PluggableVLMPlanner

plt.style.use("dark_background")

out_dir = os.path.join("ablation_eval_full", "Updated 7 models Study")
os.makedirs(out_dir, exist_ok=True)

model_path = "active_slam_ppo_robust_master.zip"
if not os.path.exists(model_path):
    model_path = "active_slam_ppo.zip"

def run_omniray_policy(env, m_path, seed, max_steps=300):
    model = PPO.load(m_path)
    obs, info = env.reset(seed=seed)
    base_env = env.env if hasattr(env, "env") else env
    gt_trajectory, coverage_history = [(base_env._robot_x, base_env._robot_y)], [info["coverage"]]
    step_count, total_distance, collisions = 0, 0.0, 0
    start_time = time.time()
    time_history = [0.0]

    while step_count < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        gt_x, gt_y = base_env._robot_x, base_env._robot_y
        total_distance += np.hypot(gt_x - gt_trajectory[-1][0], gt_y - gt_trajectory[-1][1])
        gt_trajectory.append((gt_x, gt_y))
        coverage_history.append(info["coverage"])
        time_history.append(time.time() - start_time)
        if info.get("collision", False): collisions += 1
        if terminated or truncated: break

    return {
        "name": "OmniRay (Ours - CPU AVX2)",
        "trajectory": np.array(gt_trajectory),
        "coverage": coverage_history,
        "final_coverage": coverage_history[-1],
        "total_distance": total_distance,
        "collisions": collisions,
        "steps": step_count,
        "wall_time": time.time() - start_time,
        "time_history": time_history
    }

def evaluate_suite_on_map(walls, map_name, out_filename):
    def create_base():
        e = ActiveSLAMEnv(backend="simd", num_rays=128, map_resolution=50, max_steps=300, use_slam=True, real_world_noise=True)
        e._walls = walls
        e.raycaster.set_walls(walls)
        return e

    seeds = [42, 43, 44]
    vlm_planner = PluggableVLMPlanner(model_name="DeepSeek-R1 / Qwen-2.5-VL", simulated_latency_ms=650.0)

    print(f"\n=======================================================")
    print(f"  Evaluating 7 Models on {map_name}")
    print(f"=======================================================")

    results = {}
    
    print("  [1/7] Running Random Walk...")
    results["Random Walk"] = [run_random_walk(create_base(), s) for s in seeds]
    
    print("  [2/7] Running Yamauchi (1997)...")
    results["Yamauchi (1997)"] = [run_yamauchi(create_base(), s) for s in seeds]
    
    print("  [3/7] Running RRT-Exploration (2017)...")
    results["RRT-Exploration (2017)"] = [run_rrt_exploration(create_base(), s) for s in seeds]
    
    print("  [4/7] Running Stachniss (2005)...")
    results["Stachniss (2005)"] = [run_stachniss_entropy(create_base(), s) for s in seeds]
    
    print("  [5/7] Running Shanghai AI Lab DRL (2025)...")
    results["Shanghai AI Lab DRL (2025)"] = [run_shanghai_drl_agent(create_base(), s) for s in seeds]
    
    print("  [6/7] Running DeepSeek-R1 / Qwen-2.5-VL...")
    results["DeepSeek-R1 / Qwen-2.5-VL"] = [run_vlm_agent(create_base(), vlm_planner, s) for s in seeds]
    
    print("  [7/7] Running OmniRay (Ours)...")
    omniray_runs = []
    for s in seeds:
        b_env = create_base()
        ad_env = AdaptiveActiveSLAMEnv(b_env, enable_health=True, enable_adaptive_reward=True, enable_meta=True, enable_curriculum=True, enable_continual=True)
        omniray_runs.append(run_omniray_policy(ad_env, model_path, s))
    results["OmniRay (Ours - CPU AVX2)"] = omniray_runs

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor="#090d13")
    
    # Left: Trajectories (Seed 42)
    ax1 = axes[0]
    ax1.set_facecolor("#0d1117")
    for x1, y1, x2, y2 in walls:
        ax1.plot([x1, x2], [y1, y2], color="#ff6b6b", linewidth=1.8, alpha=0.7)

    colors = {
        "Random Walk": "#7f8c8d",
        "Yamauchi (1997)": "#e74c3c",
        "RRT-Exploration (2017)": "#3498db",
        "Stachniss (2005)": "#9b59b6",
        "Shanghai AI Lab DRL (2025)": "#e67e22",
        "DeepSeek-R1 / Qwen-2.5-VL": "#f1c40f",
        "OmniRay (Ours - CPU AVX2)": "#2ecc71"
    }

    plot_order = ["Random Walk", "Yamauchi (1997)", "RRT-Exploration (2017)", "Stachniss (2005)", "Shanghai AI Lab DRL (2025)", "DeepSeek-R1 / Qwen-2.5-VL", "OmniRay (Ours - CPU AVX2)"]

    for name in plot_order:
        runs = results[name]
        traj = runs[0]["trajectory"]
        lw = 2.6 if "OmniRay" in name else 1.5
        ax1.plot(traj[:, 0], traj[:, 1], color=colors[name], linewidth=lw, label=name)

    ax1.set_title(f"Exploration Trajectories ({map_name} - Seed 42)", color="#f0f6fc", fontsize=11, fontweight="bold")
    ax1.set_xlim(-5, 105); ax1.set_ylim(-5, 105); ax1.set_aspect("equal")
    ax1.grid(color="#21262d", linestyle="--", alpha=0.6)
    ax1.tick_params(colors="#8b949e")
    ax1.legend(loc="upper left", facecolor="#161b22", edgecolor="#30363d", labelcolor="#f0f6fc", fontsize=7.5)

    # Right: Coverage vs Wall-Clock Time
    ax2 = axes[1]
    ax2.set_facecolor("#0d1117")
    for name in plot_order:
        runs = results[name]
        min_len = min(len(r["coverage"]) for r in runs)
        avg_cov = np.mean([r["coverage"][:min_len] for r in runs], axis=0) * 100
        avg_time = np.mean([r["time_history"][:min_len] for r in runs], axis=0)
        lw = 2.6 if "OmniRay" in name else 1.5
        ax2.plot(avg_time, avg_cov, color=colors[name], linewidth=lw, label=name)

    ax2.set_title(f"Coverage Rate vs. Wall-Clock Time ({map_name} — 3-Seed Mean)", color="#f0f6fc", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Wall-Clock Time (seconds)", color="#8b949e", fontsize=10)
    ax2.set_ylabel("Map Coverage (%)", color="#8b949e", fontsize=10)
    ax2.grid(color="#21262d", linestyle="--", alpha=0.6)
    ax2.tick_params(colors="#8b949e")
    ax2.legend(loc="lower right", facecolor="#161b22", edgecolor="#30363d", labelcolor="#f0f6fc", fontsize=7.5)

    plt.suptitle(f"Multi-Model Active Exploration Benchmark ({map_name})", color="#58a6ff", fontsize=13, fontweight="bold")
    plt.tight_layout()
    
    out_file = os.path.join(out_dir, out_filename)
    plt.savefig(out_file, dpi=200, facecolor="#090d13")
    plt.close()
    print(f"  [SUCCESS] Saved exact benchmark report to: {out_file}")

if __name__ == "__main__":
    evaluate_suite_on_map(get_intel_lab_walls(), "Intel Research Lab", "multi_model_intel_benchmark_7models.png")
    evaluate_suite_on_map(get_mit_stata_walls(), "MIT Stata Center", "mit_stata_multi_model_benchmark_7models.png")
