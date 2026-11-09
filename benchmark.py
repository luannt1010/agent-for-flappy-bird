"""CLI for reproducible Flappy Bird algorithm/environment benchmarks.

Examples::
    python benchmark.py --methods dqn rainbow ppo --seeds 0 1 2 3 4
    python benchmark.py --methods all --seeds 0 --train-steps 32 \
        --warmup-steps 8 --batch-size 4 --rollout-steps 8 --ppo-epochs 1 \
        --validation-interval 32 --validation-episodes 1 --test-episodes 1 \
        --max-episode-steps 40 --eval-profiles standard --device cpu

The second command is an integration smoke test, not evidence of learned skill.
"""

import argparse
from dataclasses import asdict
from datetime import datetime
import math
import os
from pathlib import Path
import platform
import sys
import time
from zoneinfo import ZoneInfo


def get_args(argv=None):
    """Parse train/evaluate CLI options; return Namespace with validated budgets.

    Pass argv as a list for testing, or None to use sys.argv. Training/test seed
    namespaces cannot overlap. AgentConfig performs hyperparameter validation.
    """
    from src.agent_registry import METHODS
    from src.experiment_config import PROFILES, TRAIN_STRATEGIES, training_setup
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("train", "evaluate"), default="train")
    parser.add_argument("--methods", nargs="+", choices=(*METHODS, "all"), default=["dqn", "rainbow", "ppo"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--train-steps", type=int, default=1_000_000, help="Maximum aggregate observed bird transitions per method/seed")
    parser.add_argument("--validation-interval", type=int, default=10_000)
    parser.add_argument("--validation-episodes", type=int, default=10)
    parser.add_argument("--test-episodes", type=int, default=100)
    parser.add_argument("--validation-seed-start", type=int, default=1_000_000)
    parser.add_argument("--test-seed-start", type=int, default=2_000_000)
    parser.add_argument("--early-stopping-episodes", type=int, default=50_000, help="0 disables; patience since best validation, counted in episodes/competitive rounds")
    parser.add_argument("--min-delta", type=float, default=0.0, help="Minimum validation reward improvement to replace checkpoint")
    parser.add_argument("--max-episode-steps", type=int, default=10_000)
    parser.add_argument("--train-profile", choices=PROFILES, default="standard",
                        help="Fixed training physics, or final curriculum endpoint")
    parser.add_argument("--train-strategy", choices=TRAIN_STRATEGIES, default=None,
                        help="Independent of learner; default fixed (legacy method aliases keep their strategy)")
    parser.add_argument("--curriculum-fraction", type=float, default=1.0,
                        help="Budget fraction to reach final physics; remaining budget stays at endpoint")
    parser.add_argument("--validation-profiles", nargs="+", choices=PROFILES, default=["standard"],
                        help="Select checkpoint by equal-weight mean reward across these fixed profiles")
    parser.add_argument("--eval-profiles", nargs="+", choices=PROFILES,
                        default=["standard", "narrow", "fast", "wind"])
    parser.add_argument("--num-agents", type=int, default=2)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--render", action="store_true", help="Show game window (headless/full speed by default)")
    parser.add_argument("--render-fps", type=int, default=30,
                        help="Maximum FPS for rendered evaluation; 30 is real time, 0 is unlimited")
    parser.add_argument("--no-progress", dest="progress", action="store_false", default=True,
                        help="Disable tqdm progress bars while keeping validation/test logs")
    parser.add_argument("--nondeterministic", dest="deterministic", action="store_false", default=True)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--log-interval", type=int, default=1000, help="Transition interval for retained loss diagnostics")
    parser.add_argument("--checkpoint", action="append", default=[], metavar="METHOD:SEED=PATH",
                        help="Repeat once per run in --mode evaluate; benchmark format only")
    parser.add_argument("--baseline-checkpoint", type=Path, help="Additionally test an existing original DQN checkpoint; reported separately as dqn_pretrained")
    parser.add_argument("--trust-legacy-checkpoint", action="store_true", help="Allow pickle loading only for your trusted original model object")
    parser.add_argument("--hyperparameters", type=Path, help="JSON object with AgentConfig overrides, optionally {'default': {...}, 'ppo': {...}, ...}")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-capacity", type=int, default=10000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--train-frequency", type=int, default=4)
    parser.add_argument("--target-update-interval", type=int, default=1000, help="Optimizer updates, not environment steps")
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--n-step", type=int, default=3)
    options = parser.parse_args(argv)
    if "all" in options.methods:
        options.methods = list(METHODS)
    options.methods = list(dict.fromkeys(options.methods))
    options.eval_profiles = list(dict.fromkeys(options.eval_profiles))
    options.validation_profiles = list(dict.fromkeys(options.validation_profiles))
    try:
        for method in options.methods:
            training_setup(method, options.train_profile, options.train_strategy, options.curriculum_fraction)
    except ValueError as error:
        parser.error(str(error))
    if len(set(options.seeds)) != len(options.seeds) or any(s < 0 or s >= 2**32 - options.num_agents for s in options.seeds):
        parser.error("Training seeds must be unique nonnegative 32-bit integers")
    for field in ("train_steps", "validation_interval", "validation_episodes", "test_episodes", "max_episode_steps", "torch_threads", "log_interval"):
        if getattr(options, field) < 1:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if options.num_agents < 2 or ("multi_agent" in options.methods and options.train_steps < options.num_agents):
        parser.error("--num-agents must be >=2 and <= multi_agent training budget")
    if options.early_stopping_episodes < 0 or not math.isfinite(options.min_delta) or options.min_delta < 0:
        parser.error("Early stopping patience and min-delta must be nonnegative")
    if options.render_fps < 0:
        parser.error("--render-fps must be nonnegative (0 disables the limit)")
    if not math.isfinite(options.curriculum_fraction) or not 0 < options.curriculum_fraction <= 1:
        parser.error("--curriculum-fraction must be in (0, 1]")
    from src.rl_common import seed_sequence
    try:
        options.validation_seeds = seed_sequence(options.validation_seed_start, options.validation_episodes)
        options.test_seeds = seed_sequence(options.test_seed_start, options.test_episodes)
    except ValueError as error:
        parser.error(str(error))
    training_seeds = {seed + player for seed in options.seeds for player in range(options.num_agents)}
    if (set(options.validation_seeds) & set(options.test_seeds)
            or training_seeds & (set(options.validation_seeds) | set(options.test_seeds))):
        parser.error("Training, validation and test seed namespaces must be disjoint")
    if options.mode == "evaluate" and not options.checkpoint:
        parser.error("--mode evaluate requires --checkpoint METHOD:SEED=PATH for every requested run")
    if options.checkpoint and options.mode != "evaluate":
        parser.error("--checkpoint is for --mode evaluate; use --baseline-checkpoint for an extra original DQN")
    return options


def resolve_configs(options):
    """Merge CLI defaults with optional JSON method overrides; return AgentConfig map.

    Input JSON is either a flat AgentConfig object or default/method objects.
    All effective hyperparameters are serialized into results and checkpoints.
    """
    import json
    from src.agent_registry import METHODS
    from src.rl_common import AgentConfig
    cli_fields = ("learning_rate", "gamma", "batch_size", "replay_capacity", "warmup_steps",
                  "train_frequency", "target_update_interval", "rollout_steps", "ppo_epochs", "minibatch_size", "n_step")
    base = {**asdict(AgentConfig()), **{key: getattr(options, key) for key in cli_fields}}
    base["epsilon_decay_steps"] = options.train_steps
    base["per_beta_steps"] = options.train_steps
    overrides = json.loads(options.hyperparameters.read_text(encoding="utf-8")) if options.hyperparameters else {}
    if not isinstance(overrides, dict):
        raise ValueError("Hyperparameters JSON must be an object")
    grouped = any(key in overrides for key in ("default", *METHODS))
    if grouped:
        unknown = set(overrides) - {"default", *METHODS}
        if unknown:
            raise ValueError(f"Unknown method overrides: {sorted(unknown)}")
        defaults = overrides.get("default", {})
    else:
        defaults = overrides
    return {method: AgentConfig(**{**base, **defaults, **(overrides.get(method, {}) if grouped else {})})
            for method in options.methods}


def checkpoint_map(specifications: list[str]) -> dict:
    """Parse METHOD:SEED=PATH into (method,seed)->Path; reject duplicates/missing files."""
    result = {}
    for specification in specifications:
        try:
            key, path = specification.split("=", 1)
            method, seed = key.rsplit(":", 1)
            key = (method, int(seed))
        except ValueError as error:
            raise ValueError("Checkpoint syntax must be METHOD:SEED=PATH") from error
        if key in result:
            raise ValueError(f"Duplicate checkpoint mapping: {key}")
        result[key] = Path(path).resolve()
        if not result[key].is_file():
            raise FileNotFoundError(result[key])
    return result


def evaluate_legacy(path: Path, options) -> dict:
    """Evaluate a trusted original DQN once; never label it as multiple training seeds.

    Accepts a bare state_dict, {'state_dict': ...}, a benchmark DQN checkpoint, or
    an original nn.Module only with explicit --trust-legacy-checkpoint opt-in.
    """
    import torch
    from src.agent_registry import make_agent
    from src.benchmarking import evaluate_agents, load_checkpoint
    from src.rl_common import AgentConfig
    payload = torch.load(path, map_location=options.device, weights_only=not options.trust_legacy_checkpoint)
    if isinstance(payload, dict) and payload.get("schema_version") == 1:
        agents, _ = load_checkpoint(path, options.device, expected_method="dqn")
    else:
        if isinstance(payload, torch.nn.Module):
            payload = payload.state_dict()
        if not isinstance(payload, dict):
            raise ValueError("Unsupported original DQN checkpoint format")
        agent = make_agent("dqn", AgentConfig(), options.device)
        agent.load_state_dict(payload.get("state_dict", payload))
        agents = [agent]
    started = time.perf_counter()
    test = evaluate_agents(agents, options.test_seeds, options.max_episode_steps, options.eval_profiles, options.render,
                           progress=options.progress, progress_description="Test dqn_pretrained",
                           render_fps=options.render_fps)
    return {"method": "dqn_pretrained", "train_seed": None, "checkpoint": str(path.resolve()),
            "cost": None, "history": [], "test": test, "test_seconds": time.perf_counter() - started,
            "training_provenance": "External checkpoint; training budget/seed unknown. Excluded from equal-budget claims."}


def main(argv=None) -> int:
    """Execute requested runs, write incremental JSON and plots; return exit code 0."""
    options = get_args(argv)
    if not options.render:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import numpy as np
    import cv2
    import pygame
    from src.provenance import source_provenance
    from src.benchmarking import (aggregate_runs, evaluate_agents, evaluate_competition, load_checkpoint,
                                  plot_results, train_run, write_json)
    from src.progress import create_progress, progress_write
    torch.set_num_threads(options.torch_threads)
    if options.device == "auto":
        options.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(options.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices: cpu, cuda or cuda:N")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    configs = resolve_configs(options)
    mapping = checkpoint_map(options.checkpoint)
    if options.mode == "evaluate":
        required = {(method, seed) for method in options.methods for seed in options.seeds}
        if set(mapping) != required:
            raise ValueError(f"Checkpoint mappings must exactly match requested method/seed pairs. Missing={required - set(mapping)}, extra={set(mapping) - required}")
    now = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh"))
    provenance = source_provenance()
    options.source_fingerprint = provenance["sha256"]
    output = options.output_dir.resolve() / now.strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(options).items()}
    report = {"schema_version": 2, "created_at": now.isoformat(), "settings": settings,
              "source_provenance": provenance,
              "runtime": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                          "opencv": cv2.__version__, "pygame": pygame.version.ver,
                          "platform": platform.platform(), "device": options.device,
                          "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
              "protocol": {"selection": "strict improvement in equal-weight mean reward over validation profiles",
                           "test_action": "deterministic argmax; epsilon=0; NoisyNet uses mean weights",
                           "budget_unit": "one observed bird transition; aggregate across competitive players",
                           "uncertainty": "sample standard deviation across independent training-seed means",
                           "survival": "episode_steps / 30, simulated seconds; not rendering speed",
                           "baseline": "original CNN/raw pixels + MSE, detached target, separate target network, clean episode reset",
                           "checkpoint": "validation selection/evaluation only; replay and complete resume state omitted",
                           "multi_agent": "independent DQN, common pipe course, survival competition; standard game rewards",
                           "early_stopping": "experimental stopping rule, not a convergence guarantee"},
              "effective_hyperparameters": {method: {} for method in options.methods},
              "runs": [], "summary": {}}
    write_json(output / "results.json", report)
    progress_write(f"Results directory: {output}")
    pairs = ((method, seed) for method in options.methods for seed in options.seeds)
    total_runs = len(options.methods) * len(options.seeds) + int(options.baseline_checkpoint is not None)
    with create_progress(total_runs, "Benchmark", "run", enabled=options.progress, leave=True) as overall:
        for method, seed in pairs:
            overall.set_postfix(method=method, seed=seed, phase=options.mode, refresh=False)
            if options.mode == "train":
                result = train_run(method, seed, configs[method], options, output)
                agents, _ = load_checkpoint(Path(result["checkpoint"]), options.device, method, seed)
            else:
                agents, metadata = load_checkpoint(mapping[(method, seed)], options.device, method, seed)
                result = {"method": method, "train_seed": seed, "checkpoint": str(mapping[(method, seed)]),
                          "training_environment": metadata.get("training_environment"),
                          "validation_profiles": metadata.get("validation_profiles"),
                          "source_fingerprint": metadata.get("source_fingerprint"),
                          "agent_config": metadata["agent_config"], "selected_steps": metadata["steps"],
                          "best_validation_reward": metadata["validation_reward"], "cost": None, "history": []}
            started = time.perf_counter()
            overall.set_postfix(method=method, seed=seed, phase="test", refresh=False)
            result["test"] = evaluate_agents(agents, options.test_seeds, options.max_episode_steps,
                                             options.eval_profiles, options.render, progress=options.progress,
                                             progress_description=f"Test {method} seed={seed}",
                                             render_fps=options.render_fps)
            if method == "multi_agent":
                result["test_competition"] = evaluate_competition(
                    agents, options.test_seeds, options.max_episode_steps, progress=options.progress,
                    progress_description=f"Test competition seed={seed}")
            result["test_seconds"] = time.perf_counter() - started
            report["effective_hyperparameters"][method][str(seed)] = result["agent_config"]
            report["runs"].append(result)
            report["summary"] = aggregate_runs(report["runs"], options.eval_profiles)
            write_json(output / "results.json", report)
            standard = options.eval_profiles[0]
            progress_write(f"[{method} seed={seed}] test/{standard}: "
                           f"score={result['test'][standard]['summary']['score']['mean']:.3f}, "
                           f"reward={result['test'][standard]['summary']['reward']['mean']:.3f}")
            overall.update(1)
            del agents
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if options.baseline_checkpoint:
            overall.set_postfix(method="dqn_pretrained", phase="test", refresh=False)
            report["runs"].append(evaluate_legacy(options.baseline_checkpoint, options))
            report["summary"] = aggregate_runs(report["runs"], options.eval_profiles)
            overall.update(1)
    report["plots"] = plot_results(report["runs"], report["summary"], output)
    write_json(output / "results.json", report)
    write_json(output / "summary.json", report["summary"])
    progress_write(f"Saved JSON and comparison plots to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
