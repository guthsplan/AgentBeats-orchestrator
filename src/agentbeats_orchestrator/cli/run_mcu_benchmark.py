from __future__ import annotations

import argparse
import json

from agentbeats_orchestrator.actor.openha_adapter import OpenHAActorAdapter
from agentbeats_orchestrator.benchmark.runner import run_benchmark
from agentbeats_orchestrator.evaluator.heuristic import HeuristicEvaluator
from agentbeats_orchestrator.planner.trivial import TrivialPlanner
from agentbeats_orchestrator.runtime.loop import PurpleRuntime
from agentbeats_orchestrator.transition.rule_based import RuleBasedTransitionPolicy


def build_openha_agent(args):
    from openagents.agents.openha import OpenHA

    return OpenHA(
        model_path=args.model_path,
        output_mode=args.output_mode,
        output_format=args.output_format,
        vlm_client_mode=args.vlm_client_mode,
        model_url=args.model_url,
        model_id=args.model_id,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        grounding_policy_path=args.grounding_policy_path,
        motion_policy_path=args.motion_policy_path,
        sam_path=args.sam_path,
        grounding_inference_interval=args.grounding_inference_interval,
        motion_inference_interval=args.motion_inference_interval,
        maximum_history_length=args.maximum_history_length,
        raw_action_type=args.raw_action_type,
    )


def make_runtime_factory(args):
    def _factory():
        openha_agent = build_openha_agent(args)
        return PurpleRuntime(
            planner=TrivialPlanner(),
            actor=OpenHAActorAdapter(openha_agent=openha_agent),
            evaluator=HeuristicEvaluator(),
            transition_policy=RuleBasedTransitionPolicy(),
        )

    return _factory


def main():
    parser = argparse.ArgumentParser(description="Run AgentBeats MCU benchmark")
    parser.add_argument("--tasks-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--category", action="append", default=None)
    parser.add_argument("--max-steps", type=int, default=600)

    parser.add_argument("--vlm-client-mode", default="online")
    parser.add_argument("--output-mode", default="text_action")
    parser.add_argument("--output-format", default="text_action")
    parser.add_argument("--raw-action-type", default="text")

    parser.add_argument("--model-id", default="CrossAgent-qwen2vl-7b")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-url", default="http://localhost:11000/v1")
    parser.add_argument("--api-key", default="EMPTY")

    parser.add_argument("--sam-path", required=True)
    parser.add_argument("--grounding-policy-path", required=True)
    parser.add_argument("--motion-policy-path", required=True)

    parser.add_argument("--grounding-inference-interval", type=int, default=4)
    parser.add_argument("--motion-inference-interval", type=int, default=4)
    parser.add_argument("--maximum-history-length", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max-tokens", type=int, default=512)

    args = parser.parse_args()
    results = run_benchmark(
        tasks_dir=args.tasks_dir,
        output_dir=args.output_dir,
        runtime_factory=make_runtime_factory(args),
        categories=args.category,
        max_steps=args.max_steps,
        verbose=True,
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
