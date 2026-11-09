"""Training, held-out evaluation and statistical reporting for Flappy Bird.

Validation selects checkpoints using mean undiscounted reward; test courses are
never consulted during selection. One budget unit is one observed bird action,
including in multi-agent runs. Time limits truncate and bootstrap; deaths terminate.
"""

from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
import json
import random
import time

import numpy as np
import pygame
import torch

from src.agent_registry import make_agent
from src.benchmark_env import BenchmarkEnv
from src.experiment_config import make_training_env, profile_layout, profile_physics, training_setup
from src.progress import create_progress, progress_write
from src.provenance import source_provenance
from src.rl_common import AgentConfig, seed_everything


@contextmanager
def isolated_random_state():
    """Restore Python/NumPy/Torch RNGs after validation (including CUDA RNGs).

    No inputs/outputs. Validation cannot alter subsequent exploration, noisy
    layers or minibatch ordering merely because its frequency changed.
    """
    py_state, np_state = random.getstate(), np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def write_json(path: Path, payload) -> None:
    """Atomically write JSON data to path; reject NaN/Infinity instead of hiding errors."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def summarize_episodes(episodes: list[dict]) -> dict:
    """Return episode means and sample SDs (ddof=1); one observation gives SD=0.

    These are within-run descriptive statistics, not uncertainty across training
    seeds. Fields include score, reward, steps and simulated survival seconds.
    """
    result = {"episodes": len(episodes)}
    for key in ("score", "reward", "episode_steps", "survival_seconds"):
        values = np.array([row[key] for row in episodes], dtype=np.float64)
        result[key] = {"mean": float(values.mean()) if len(values) else 0.0,
                       "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
    result["truncation_rate"] = float(np.mean([r["truncated"] for r in episodes])) if episodes else 0.0
    for threshold in (10, 50):
        values = np.array([r["score"] >= threshold for r in episodes], dtype=np.float64)
        result[f"success_rate_{threshold}"] = {
            "mean": float(values.mean()) if len(values) else 0.0,
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return result


def evaluate_agents(agents: list, seeds: list[int], max_episode_steps: int,
                    profiles=("standard",), render=False, *, progress=False,
                    progress_description="Evaluation", progress_position=1,
                    render_fps: int = 30) -> dict:
    """Evaluate deterministic policies on identical held-out courses per profile.

    Input agents follow the common protocol. Every player gets every episode
    seed; output contains raw episode records and per-profile statistics. For
    competitive training, individual policy scores are pooled with player_id.
    No transition is observed and no optimizer is called during evaluation.
    Optional progress counts completed episodes across all players/profiles;
    its display settings do not enter the returned metrics.
    A rendered demo is capped at render_fps (30 real-time game steps/second).
    Zero disables the cap. Headless evaluation never creates or ticks a clock;
    pacing does not change physics, observations, actions, or episode metrics.
    """
    if not isinstance(render_fps, int) or render_fps < 0:
        raise ValueError("render_fps must be a nonnegative integer")
    clock = pygame.time.Clock() if render and render_fps else None
    results = {}
    with create_progress(len(agents) * len(seeds) * len(profiles), progress_description,
                         "episode", enabled=progress, position=progress_position) as bar, \
            isolated_random_state(), torch.inference_mode():
        for profile in profiles:
            bar.set_postfix(profile=profile, refresh=False)
            env = BenchmarkEnv(max_episode_steps=max_episode_steps, render=render,
                               physics=profile_physics(profile), layout=profile_layout(profile))
            episodes = []
            try:
                for player_id, agent in enumerate(agents):
                    bar.set_postfix(profile=profile, player=player_id, refresh=False)
                    for seed in seeds:
                        state = env.reset(seed=seed)
                        total_reward, steps = 0.0, 0
                        while True:
                            state, reward, terminated, truncated, info = env.step(agent.act(state, explore=False))
                            if clock is not None:
                                clock.tick(render_fps)
                            total_reward += float(reward)
                            steps += 1
                            if terminated or truncated:
                                break
                        episodes.append({"seed": seed, "player_id": player_id,
                                         "score": int(info["score"]), "reward": total_reward,
                                         "episode_steps": steps, "survival_seconds": steps / 30.0,
                                         "terminated": bool(terminated), "truncated": bool(truncated)})
                        bar.set_postfix(profile=profile, player=player_id,
                                        score=int(info["score"]), reward=f"{total_reward:.3f}", refresh=False)
                        bar.update(1)
            finally:
                env.close()
            results[profile] = {"physics": asdict(profile_physics(profile)),
                                "layout": asdict(profile_layout(profile)),
                                "summary": summarize_episodes(episodes), "episodes": episodes}
    return results


def evaluate_competition(agents: list, seeds: list[int], max_episode_steps: int, *,
                         progress=False, progress_description="Test competition",
                         progress_position=1) -> dict:
    """Race trained policies on held-out common courses; return fractional win rates.

    Every tied winner receives 1/number_of_winners per course. Individual death
    freezes that bird, while the shared course continues until all birds finish.
    This supplements the individually evaluated test scores, without training.
    Optional progress counts shared races, not the number of participating birds.
    """
    from src.multi_agent import CompetitiveFlappyBird
    env = CompetitiveFlappyBird(num_agents=len(agents), max_episode_steps=max_episode_steps)
    wins = np.zeros(len(agents), dtype=np.float64)
    episodes = []
    try:
        with create_progress(len(seeds), progress_description, "race", enabled=progress,
                             position=progress_position) as bar, \
                isolated_random_state(), torch.inference_mode():
            for seed in seeds:
                states = env.reset(seed=seed)
                while True:
                    alive = env.alive
                    actions = [a.act(s, explore=False) if live else 0
                               for a, s, live in zip(agents, states, alive)]
                    states, _, _, _, info = env.step(actions)
                    if info["finished"]:
                        break
                for winner in info["winner_ids"]:
                    wins[winner] += 1.0 / len(info["winner_ids"])
                episodes.append({"seed": seed, "winner_ids": info["winner_ids"],
                                 "ranking": info["ranking"], "agents": info["agents"]})
                bar.set_postfix(winners=",".join(map(str, info["winner_ids"])), refresh=False)
                bar.update(1)
    finally:
        env.close()
    return {"episodes": episodes, "fractional_win_rates": (wins / len(seeds)).tolist(),
            "ranking_rule": "survival steps, then pipes passed; exact ties share a win"}


def save_checkpoint(path: Path, agents: list, method: str, seed: int,
                    config: AgentConfig, steps: int, validation_reward: float,
                    training_environment=None, validation_profiles=None, source_fingerprint=None) -> None:
    """Save model/optimizer selection state atomically; replay is deliberately excluded.

    Returned file supports evaluation, not bit-exact continuation of training.
    Inputs include the selected validation mean reward and aggregate action count.
    """
    payload = {"schema_version": 1, "method": method, "train_seed": seed,
               "agent_config": asdict(config), "steps": steps,
               "validation_reward": validation_reward,
               "training_environment": training_environment,
               "validation_profiles": validation_profiles,
               "source_fingerprint": source_fingerprint,
               "agents": [agent.state_dict() for agent in agents],
               "purpose": "validation-selected evaluation checkpoint; replay not saved"}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, device="cpu", expected_method=None, expected_seed=None):
    """Safely load a benchmark checkpoint and return (agents, metadata).

    Uses weights_only=True. Method/seed checks prevent accidental cross-labelling
    in CLI evaluations. Legacy pickled models use a separate explicit opt-in.
    """
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("schema_version") != 1 or "agents" not in payload:
        raise ValueError(f"Not a benchmark checkpoint: {path}")
    if expected_method is not None and payload["method"] != expected_method:
        raise ValueError(f"Checkpoint method mismatch: {payload['method']} != {expected_method}")
    if expected_seed is not None and payload["train_seed"] != expected_seed:
        raise ValueError("Checkpoint training seed does not match requested run")
    config = AgentConfig(**payload["agent_config"])
    agents = []
    for index, state in enumerate(payload["agents"]):
        agent = make_agent(payload["method"], config, device, payload["train_seed"] + index)
        agent.load_state_dict(state)
        agents.append(agent)
    return agents, payload


def _synchronize(device) -> None:
    """Wait for CUDA kernels before wall-clock accounting; CPU is a no-op."""
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def train_run(method: str, seed: int, config: AgentConfig, options, run_dir: Path) -> dict:
    """Train one independent seed under a maximum aggregate action budget.

    Validation occurs every validation_interval observed transitions and at the
    final budget boundary. Early stopping counts completed single-player episodes
    (competitive rounds for multi-agent), not gradient updates. Return selected
    checkpoint path, learning curves and measured costs; never access test seeds.
    """
    seed_everything(seed, options.deterministic)
    fingerprint = getattr(options, "source_fingerprint", None) or source_provenance()["sha256"]
    count = options.num_agents if method == "multi_agent" else 1
    agents = [make_agent(method, config, options.device, seed + i) for i in range(count)]
    setup = training_setup(method, options.train_profile, options.train_strategy, options.curriculum_fraction)
    if method == "multi_agent":
        from src.multi_agent import CompetitiveFlappyBird
        env = CompetitiveFlappyBird(num_agents=count, seed=seed,
                                   max_episode_steps=options.max_episode_steps, render=options.render)
    else:
        env = make_training_env(setup, seed=seed, max_episode_steps=options.max_episode_steps,
                                render=options.render)
    states = env.reset(seed=seed, progress=0.0)
    if count == 1:
        states = [states]
    checkpoint = run_dir / method / f"seed_{seed}" / "best.pt"
    history, losses, training_episodes = [], [], []
    returns = [0.0] * count
    episode_lengths = [0] * count
    steps = completed_episodes = best_episode = 0
    best_reward = -float("inf")
    validation_seconds = optimization_seconds = environment_seconds = 0.0
    validation_interval = options.validation_interval
    next_validation = validation_interval
    start_time = time.perf_counter()
    stop_reason = "budget"
    selected_steps = 0
    last_validated_steps = -1
    competition = []
    show_progress = getattr(options, "progress", False)
    bar = create_progress(options.train_steps, f"Train {method} seed={seed}", "step",
                          enabled=show_progress, position=1)
    progress_metrics = {"episodes": 0}
    next_progress_status = 0
    status_interval = max(1, min(options.log_interval, 100))

    def validate():
        """Evaluate validation seeds, select strictly better reward, record curve."""
        nonlocal best_reward, best_episode, selected_steps, validation_seconds, last_validated_steps
        _synchronize(options.device)
        started = time.perf_counter()
        metrics = evaluate_agents(agents, options.validation_seeds, options.max_episode_steps,
                                  profiles=options.validation_profiles,
                                  progress=show_progress,
                                  progress_description=f"Validation {method} seed={seed}",
                                  progress_position=2)
        reward = float(np.mean([block["summary"]["reward"]["mean"] for block in metrics.values()]))
        # Equal episode counts per profile make pooling equivalent to equal weights.
        validation = (next(iter(metrics.values()))["summary"] if len(metrics) == 1 else
                      summarize_episodes([row for block in metrics.values() for row in block["episodes"]]))
        history.append({"steps": steps, "completed_episodes": completed_episodes,
                        "validation": validation,
                        "validation_by_profile": {p: b["summary"] for p, b in metrics.items()}})
        if reward > best_reward + options.min_delta:
            best_reward, best_episode, selected_steps = reward, completed_episodes, steps
            save_checkpoint(checkpoint, agents, method, seed, config, steps, reward,
                            setup, options.validation_profiles, fingerprint)
        validation_seconds += time.perf_counter() - started
        history[-1]["training_seconds"] = time.perf_counter() - start_time - validation_seconds
        last_validated_steps = steps
        progress_metrics.update(val_reward=f"{reward:.3f}", best=f"{best_reward:.3f}")
        bar.set_postfix(progress_metrics, refresh=False)
        progress_write(f"[{method} seed={seed}] steps={steps}/{options.train_steps} "
                       f"val_reward={reward:.3f} best={best_reward:.3f}")
        write_json(checkpoint.parent / "history.json", {"validation": history, "updates": losses,
                                                       "training_episodes": training_episodes})

    try:
        while steps < options.train_steps:
            alive = list(env.alive) if count > 1 else [True]
            active = [i for i in range(count) if alive[i]]
            if steps + len(active) > options.train_steps:
                break
            _synchronize(options.device)
            started = time.perf_counter()
            actions = [agents[i].act(states[i], explore=True) if alive[i] else 0 for i in range(count)]
            if count == 1:
                obs, reward, terminal, truncated, info = env.step(actions[0])
                next_states, rewards, terminals, truncations = [obs], [reward], [terminal], [truncated]
                infos = [info]
            else:
                next_states, rewards, terminals, truncations, info = env.step(actions)
                infos = info["agents"]
            environment_seconds += time.perf_counter() - started
            started = time.perf_counter()
            for i in active:
                agents[i].observe(states[i], actions[i], rewards[i], next_states[i], terminals[i], truncations[i])
                diagnostics = agents[i].update()
                if diagnostics:
                    for loss_key in ("loss", "critic_loss", "policy_loss"):
                        if loss_key in diagnostics:
                            progress_metrics["loss"] = f"{diagnostics[loss_key]:.4g}"
                            break
                    if "epsilon" in diagnostics:
                        progress_metrics["epsilon"] = f"{diagnostics['epsilon']:.4g}"
                steps += 1
                returns[i] += float(rewards[i])
                episode_lengths[i] += 1
                if diagnostics and (not losses or steps - losses[-1]["steps"] >= options.log_interval):
                    losses.append({"steps": steps, "player_id": i,
                                   **{k: float(v) for k, v in diagnostics.items()}})
                if terminals[i] or truncations[i]:
                    training_episodes.append({"steps": steps, "player_id": i, "reward": returns[i],
                                              "score": int(infos[i]["score"]),
                                              "episode_steps": episode_lengths[i],
                                              "terminated": bool(terminals[i]), "truncated": bool(truncations[i])})
                    returns[i], episode_lengths[i] = 0.0, 0
            _synchronize(options.device)
            optimization_seconds += time.perf_counter() - started
            states = next_states
            finished = bool(terminals[0] or truncations[0]) if count == 1 else bool(info["finished"])
            if finished:
                completed_episodes += 1
                if count > 1:
                    competition.append({"round": completed_episodes, "winner_ids": info["winner_ids"],
                                        "ranking": info["ranking"]})
                fraction = setup["curriculum_fraction"] or 1.0
                states = env.reset(progress=min(1.0, steps / (options.train_steps * fraction)))
                if count == 1:
                    states = [states]
            if steps >= next_progress_status or finished:
                progress_metrics["episodes"] = completed_episodes
                bar.set_postfix(progress_metrics, refresh=False)
                next_progress_status = steps + status_interval
            bar.update(len(active))
            # All learners get one final validation, after finish() consumes the
            # remaining rollout. PPO must not get an extra pre-flush candidate.
            if steps >= next_validation and steps < options.train_steps:
                validate()
                next_validation = (steps // validation_interval + 1) * validation_interval
                if options.early_stopping_episodes > 0 and completed_episodes - best_episode >= options.early_stopping_episodes:
                    stop_reason = "early_stopping"
                    break
        started = time.perf_counter()
        flushed_updates = False
        for agent in agents:
            finish = getattr(agent, "finish", None)
            if finish is not None:
                diagnostics = finish()
                if diagnostics:
                    flushed_updates = True
                    losses.append({"steps": steps, "phase": "final_rollout",
                                   **{k: float(v) for k, v in diagnostics.items()}})
        _synchronize(options.device)
        optimization_seconds += time.perf_counter() - started
        # On-policy finish may change weights after the last scheduled validation.
        if last_validated_steps != steps or flushed_updates:
            validate()
    except BaseException as error:
        stop_reason = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        raise
    finally:
        # Account for actions observed before an exception interrupts the joint step.
        if show_progress:
            bar.update(steps - bar.n)
            progress_metrics.update(episodes=completed_episodes, stop=stop_reason)
            bar.set_postfix(progress_metrics, refresh=False)
        bar.close()
        env.close()
    _synchronize(options.device)
    elapsed = time.perf_counter() - start_time
    result = {"method": method, "train_seed": seed, "agent_config": asdict(config),
              "training_environment": setup, "validation_profiles": options.validation_profiles,
              "source_fingerprint": fingerprint,
              "checkpoint": str(checkpoint.resolve()), "selected_steps": selected_steps,
              "best_validation_reward": best_reward, "actual_train_steps": steps,
              "max_train_steps": options.train_steps, "completed_episodes": completed_episodes,
              "num_players": count, "stop_reason": stop_reason,
              "cost": {"total_seconds": elapsed, "validation_seconds": validation_seconds,
                       "training_seconds": elapsed - validation_seconds,
                       "environment_and_action_seconds": environment_seconds,
                       "optimization_seconds": optimization_seconds,
                       "transitions_per_training_second": steps / max(elapsed - validation_seconds, 1e-9),
                       "trainable_parameters": [trainable_parameter_count(a) for a in agents]},
              "history": history, "competition": competition}
    result["first_validation_to_score"] = {}
    for threshold in (10, 50):
        first = next((row for row in history if row["validation"]["score"]["mean"] >= threshold), None)
        result["first_validation_to_score"][str(threshold)] = (
            {"steps": first["steps"], "training_seconds": first["training_seconds"]} if first else None)
    write_json(checkpoint.parent / "train_summary.json", result)
    return result


def trainable_parameter_count(agent) -> int:
    """Count unique trainable weights across all agent networks, excluding targets.

    SAC includes its actor and both critics. Shared policy/value encoders count
    once. Frozen target networks contribute zero, regardless of alias names.
    """
    parameters = {}
    for module in vars(agent).values():
        if isinstance(module, torch.nn.Module):
            parameters.update({id(p): p.numel() for p in module.parameters() if p.requires_grad})
    return sum(parameters.values())


def aggregate_runs(runs: list[dict], profiles: list[str], group_key="method") -> dict:
    """Aggregate each run's mean over independent training seeds, never episodes.

    For n seeds: mean=sum(x)/n; sample SD=sqrt(sum((x-mean)^2)/(n-1)).
    A single seed returns std=0 and n=1; it provides no between-seed estimate.
    """
    result = {}
    for method in dict.fromkeys(r[group_key] for r in runs):
        selected = [r for r in runs if r[group_key] == method]
        environments = {json.dumps(r.get("training_environment"), sort_keys=True) for r in selected}
        validations = {json.dumps(r.get("validation_profiles")) for r in selected}
        fingerprints = {r.get("source_fingerprint") for r in selected}
        if len(environments) > 1 or len(validations) > 1 or len(fingerprints) > 1:
            raise ValueError(f"Cannot pool different training/validation setups in group {method}")
        seeds = [r["train_seed"] for r in selected if r.get("train_seed") is not None]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate training seed in group {method}; runs are not independent")
        metrics = {"n_runs": len(selected),
                   "n_train_seeds": len([r for r in selected if r.get("train_seed") is not None]),
                   "profiles": {}}
        for profile in profiles:
            block = {}
            keys = ["score", "reward", "episode_steps", "survival_seconds"]
            keys.extend(key for key in ("success_rate_10", "success_rate_50")
                        if all(key in r["test"][profile]["summary"] for r in selected))
            for key in keys:
                values = np.array([r["test"][profile]["summary"][key]["mean"] for r in selected])
                block[key] = {"mean": float(values.mean()),
                              "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                              "per_seed_means": values.tolist()}
            truncation = np.array([r["test"][profile]["summary"]["truncation_rate"] for r in selected])
            block["truncation_rate"] = {"mean": float(truncation.mean()),
                                        "std": float(truncation.std(ddof=1)) if len(truncation) > 1 else 0.0,
                                        "per_seed_means": truncation.tolist()}
            metrics["profiles"][profile] = block
        costs = [r["cost"]["training_seconds"] for r in selected if r.get("cost")]
        metrics["training_seconds"] = ({"mean": float(np.mean(costs)),
                                        "std": float(np.std(costs, ddof=1)) if len(costs) > 1 else 0.0}
                                       if costs else None)
        result[method] = metrics
    return result


def plot_results(runs: list[dict], summary: dict, output: Path, group_key="method", *,
                 learning_curves=True) -> list[str]:
    """Save PNG/PDF comparison figures and validation learning curves.

    Error bars are sample SD across training-seed means, not standard errors.
    Curves use recorded validation means; test data never enter learning curves.
    Return absolute paths to generated artifacts; use the noninteractive backend.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = list(summary)
    horizontal = group_key != "method"  # Long algorithm/strategy/profile labels.
    profiles = list(summary[methods[0]]["profiles"])
    paths = []
    for profile in profiles:
        size = (max(18, len(methods) * 1.5), max(9, len(methods) * 0.7)) if horizontal else (max(12, len(methods) * 1.1), 9)
        fig, axes = plt.subplots(2, 2, figsize=size, constrained_layout=True)
        for axis, metric, label in zip(axes.flat,
                ("score", "reward", "survival_seconds", "training_seconds"),
                ("Pipes passed", "Episode reward", "Survival (simulated seconds)", "Training wall time (seconds)")):
            blocks = [summary[m]["training_seconds"] if metric == "training_seconds"
                      else summary[m]["profiles"][profile][metric] for m in methods]
            indices = [i for i, b in enumerate(blocks) if b is not None]
            means, errors = [blocks[i]["mean"] for i in indices], [blocks[i]["std"] for i in indices]
            if horizontal:
                axis.barh(indices, means, xerr=errors, capsize=3)
                axis.set_yticks(range(len(methods)), methods, fontsize=9)
                axis.invert_yaxis()
                axis.set_xlabel(label)
            else:
                axis.bar(indices, means, yerr=errors, capsize=3)
                axis.set_xticks(range(len(methods)), methods, rotation=40, ha="right")
                axis.set_ylabel(label)
            axis.grid(axis="x" if horizontal else "y", alpha=0.2)
        fig.suptitle(f"Flappy Bird: {profile} | error bars = SD across training seeds")
        for extension in ("png", "pdf"):
            path = output / f"comparison_{profile}.{extension}"
            fig.savefig(path, dpi=160)
            paths.append(str(path.resolve()))
        plt.close(fig)
    if not learning_curves:
        return paths
    fig, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for method in methods:
        histories = [r["history"] for r in runs if r[group_key] == method and r.get("history")]
        if not histories:
            continue
        common = sorted(set.intersection(*({h["steps"] for h in rows} for rows in histories)))
        if common:
            curves = np.array([[next(h for h in reversed(rows) if h["steps"] == step)["validation"]["reward"]["mean"]
                                for step in common] for rows in histories])
            mean = curves.mean(axis=0)
            std = curves.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean)
            line, = axis.plot(common, mean, label=method)
            axis.fill_between(common, mean - std, mean + std, color=line.get_color(), alpha=0.12)
        else:
            for i, rows in enumerate(histories):
                axis.plot([h["steps"] for h in rows], [h["validation"]["reward"]["mean"] for h in rows],
                          label=method if i == 0 else None, alpha=0.6)
    axis.set(xlabel="Aggregate bird transitions", ylabel="Validation mean episode reward",
             title="Validation learning curves (mean ± sample SD at common checkpoints)")
    if axis.lines:
        axis.legend(ncol=2)
    axis.grid(alpha=0.2)
    path = output / "learning_curves.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path.resolve()))
    return paths
