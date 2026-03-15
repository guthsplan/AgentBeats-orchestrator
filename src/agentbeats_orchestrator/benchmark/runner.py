from __future__ import annotations

from pathlib import Path

from agentbeats_orchestrator.benchmark.mcu_env import (
    env_init_mcu,
    list_mcu_tasks,
    save_episode_results,
)
from agentbeats_orchestrator.benchmark.reporter import save_benchmark_results


def run_single_task(
    yaml_path: str,
    runtime,
    rollout_path: str,
    obs_size: tuple[int, int] = (224, 224),
    max_steps: int = 600,
    verbose: bool = False,
):
    env, task_config = env_init_mcu(
        yaml_path=yaml_path,
        rollout_path=rollout_path,
        obs_size=obs_size,
        max_steps=max_steps,
    )
    task_text = task_config["text"]
    state = runtime.initialize(task_text)

    obs, info = env.reset()
    done = False
    step = 0
    total_reward = 0.0
    steps: list[dict] = []

    while not done and step < max_steps:
        env_action = runtime.act(obs["image"], state)
        obs, reward, terminated, truncated, info = env.step(env_action)
        done = terminated or truncated
        total_reward += reward
        transition, state = runtime.update(
            frame=obs["image"],
            reward=float(reward),
            done=done,
            state=state,
        )
        step += 1
        steps.append(
            {
                "step": step,
                "reward": float(reward),
                "decision": transition.decision.value,
                "subgoal_index": state.current_subgoal_index,
                "subgoal": state.current_subgoal.description,
            }
        )
        if verbose:
            print(
                f"step={step} reward={reward:.2f} total={total_reward:.2f} "
                f"decision={transition.decision.value}"
            )

    env.close()
    result = {
        "task_name": Path(yaml_path).stem,
        "task_text": task_text,
        "steps_taken": step,
        "total_reward": float(total_reward),
        "success": total_reward > 0,
        "steps": steps,
    }
    save_episode_results(rollout_path, result)
    return result


def run_benchmark(
    tasks_dir: str,
    output_dir: str,
    runtime_factory,
    categories: list[str] | None = None,
    obs_size: tuple[int, int] = (224, 224),
    max_steps: int = 600,
    verbose: bool = True,
):
    tasks_root = Path(tasks_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if categories is None:
        categories = [path.name for path in tasks_root.iterdir() if path.is_dir()]

    results = {
        "categories": {},
        "total_tasks": 0,
        "successful_tasks": 0,
        "total_reward": 0.0,
    }

    for category in categories:
        category_runs = []
        for yaml_path in list_mcu_tasks(str(tasks_root), category):
            runtime = runtime_factory()
            task_name = Path(yaml_path).stem
            task_output_dir = output_root / category / task_name
            result = run_single_task(
                yaml_path=yaml_path,
                runtime=runtime,
                rollout_path=str(task_output_dir),
                obs_size=obs_size,
                max_steps=max_steps,
                verbose=verbose,
            )
            category_runs.append(result)
            results["total_tasks"] += 1
            results["successful_tasks"] += int(result["success"])
            results["total_reward"] += result["total_reward"]

        if category_runs:
            results["categories"][category] = {
                "num_tasks": len(category_runs),
                "successful_tasks": sum(int(run["success"]) for run in category_runs),
                "avg_reward": sum(run["total_reward"] for run in category_runs) / len(category_runs),
                "success_rate": sum(int(run["success"]) for run in category_runs) / len(category_runs),
                "tasks": category_runs,
            }

    if results["total_tasks"] > 0:
        results["overall_success_rate"] = results["successful_tasks"] / results["total_tasks"]
        results["avg_reward"] = results["total_reward"] / results["total_tasks"]
    else:
        results["overall_success_rate"] = 0.0
        results["avg_reward"] = 0.0

    save_benchmark_results(output_dir, results)
    return results
