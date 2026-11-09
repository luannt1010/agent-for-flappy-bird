"""Run declared experiments by calling benchmark.main; never duplicate RL loops.

Examples:
    python benchmark_suite.py --config experiments/full_matrix.json --dry-run
    python benchmark_suite.py --config experiments/full_matrix.json --smoke --device cpu
    python benchmark_suite.py --config experiments/full_matrix.json --device auto
    python benchmark_suite.py --collect benchmark_results/suites/<timestamp>
"""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import shlex
from zoneinfo import ZoneInfo

import benchmark


def cli_arguments(settings: dict) -> list[str]:
    """Translate JSON option names to the existing benchmark CLI (no new defaults)."""
    arguments = []
    for key, value in settings.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                arguments.append(flag)
        elif isinstance(value, list):
            arguments.extend([flag, *map(str, value)])
        else:
            arguments.extend([flag, str(value)])
    return arguments


def experiment_definitions(config: dict) -> list[dict]:
    """Expand a compact matrix into the same explicit experiments as legacy JSON.

    The two formats share every later validation and execution step. A matrix
    declares learners once, then varies only the training profile.
    """
    if set(config) == {"description", "common", "experiments"}:
        definitions = config["experiments"]
    elif set(config) == {"description", "common", "matrix"}:
        matrix = config["matrix"]
        if set(matrix) != {"methods", "train_profiles", "train_strategy"}:
            raise ValueError("Matrix needs methods, train_profiles and train_strategy")
        definitions = [{"id": f"{matrix['train_strategy']}_{profile}",
                        "methods": matrix["methods"], "train_profile": profile,
                        "train_strategy": matrix["train_strategy"]}
                       for profile in matrix["train_profiles"]]
    else:
        raise ValueError("Suite JSON needs description, common and exactly one of experiments or matrix")
    if not definitions:
        raise ValueError("Suite must declare at least one experiment")
    return definitions


def build_plan(config: dict, *, smoke=False, device=None, seeds=None,
               train_steps=None, experiments=None) -> list[dict]:
    """Resolve and validate every experiment before starting any training.

    validation_profiles='train' selects the current training profile; an
    explicit list preserves a common validation criterion for older suites.
    """
    definitions = experiment_definitions(config)
    common = dict(config["common"])
    # Environment/model choices belong to each experiment, never hidden in common.
    forbidden = {"methods", "mode", "checkpoint", "baseline_checkpoint", "output_dir",
                 "train_profile", "train_strategy"}
    if forbidden & common.keys():
        raise ValueError(f"Options must be declared per experiment or by the runner: {forbidden & common.keys()}")
    overrides = {"device": device, "seeds": seeds, "train_steps": train_steps}
    common.update({key: value for key, value in overrides.items() if value is not None})
    if smoke:
        common.update(seeds=[0], train_steps=32, validation_interval=32,
                      validation_episodes=1, test_episodes=1, max_episode_steps=40,
                      warmup_steps=8, batch_size=4, replay_capacity=64,
                      rollout_steps=8, ppo_epochs=1, minibatch_size=4, no_progress=True)
    plan, seen = [], set()
    for experiment in definitions:
        if set(experiment) != {"id", "methods", "train_profile", "train_strategy"}:
            raise ValueError("Each experiment needs id, methods, train_profile and train_strategy")
        name = experiment["id"]
        if not isinstance(name, str) or not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in name):
            raise ValueError("Experiment id must use lowercase letters, numbers, _ or -")
        if name in seen:
            raise ValueError(f"Duplicate experiment id: {name}")
        seen.add(name)
        settings = {**common, **{k: v for k, v in experiment.items() if k != "id"}}
        if settings.get("validation_profiles") == "train":
            settings["validation_profiles"] = [settings["train_profile"]]
        arguments = cli_arguments(settings)
        options = benchmark.get_args(arguments)
        benchmark.resolve_configs(options)
        plan.append({"id": name, "arguments": arguments,
                     "runs": len(options.methods) * len(options.seeds),
                     "settings": settings})
    if experiments:
        if set(experiments) - seen:
            raise ValueError(f"Unknown experiments: {set(experiments) - seen}")
        plan = [row for row in plan if row["id"] in experiments]
    return plan


def collect_reports(reports: list[Path], output: Path) -> None:
    """Reuse existing aggregation/plots, keeping each training setup separate.

    Refuse incompatible protocols or duplicate seed runs rather than presenting
    them as independent evidence. Can also collect completed suites after an
    interruption without running training again.
    """
    from src.benchmarking import aggregate_runs, plot_results, summarize_episodes, write_json
    runs, seen, reference = [], set(), None
    test_declarations = None
    protocol_fields = ("train_steps", "validation_interval", "validation_episodes",
                       "test_episodes", "validation_seed_start", "test_seed_start",
                       "max_episode_steps", "eval_profiles",
                       "early_stopping_episodes", "min_delta", "deterministic", "torch_threads")
    for path in reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        if not report.get("source_provenance", {}).get("sha256"):
            raise ValueError(f"Missing source provenance; keep pre-audit reports separate: {path}")
        protocol = {key: report["settings"][key] for key in protocol_fields}
        protocol["runtime"] = report["runtime"]
        protocol["source_fingerprint"] = report["source_provenance"]["sha256"]
        if reference is not None and protocol != reference:
            raise ValueError(f"Cannot combine different budgets/evaluation protocols: {path}")
        reference = protocol
        for original in report["runs"]:
            row = dict(original)
            # Profile names alone cannot establish that tests used the same
            # physics/layout. Verify declarations, episode identities and cached
            # summaries before pooling any numbers.
            declarations = {profile: {key: row["test"][profile].get(key) for key in ("physics", "layout")}
                            for profile in report["settings"]["eval_profiles"]}
            if test_declarations is not None and declarations != test_declarations:
                raise ValueError(f"Different test physics/layout in {path}")
            test_declarations = declarations
            expected_seeds = report["settings"]["test_seeds"]
            expected_pairs = {(player, seed) for player in range(row.get("num_players", 1)) for seed in expected_seeds}
            for profile, block in row["test"].items():
                episodes = block["episodes"]
                pairs = [(episode["player_id"], episode["seed"]) for episode in episodes]
                if len(pairs) != len(expected_pairs) or set(pairs) != expected_pairs:
                    raise ValueError(f"Missing/duplicate/unexpected test episodes for {profile} in {path}")
                if block["summary"] != summarize_episodes(episodes):
                    raise ValueError(f"Test summary disagrees with raw episodes for {profile} in {path}")
            setup = row["training_environment"]
            row["comparison_group"] = f"{row['method']}/{setup['strategy']}/{setup['profile']}"
            identity = (row["comparison_group"], row["train_seed"])
            if identity in seen:
                raise ValueError(f"Duplicate experiment/seed: {identity}")
            seen.add(identity)
            runs.append(row)
    if not runs:
        raise ValueError("No completed runs to collect")
    # Different hyperparameters within a group must not masquerade as seed variance.
    for group in {row["comparison_group"] for row in runs}:
        configs = {json.dumps(row["agent_config"], sort_keys=True) for row in runs
                   if row["comparison_group"] == group}
        if len(configs) != 1:
            raise ValueError(f"Different learner configurations in group {group}")
    output.mkdir(parents=True, exist_ok=True)
    profiles = reference["eval_profiles"]
    summary = aggregate_runs(runs, profiles, group_key="comparison_group")
    write_json(output / "results.json", {"source_reports": [str(p.resolve()) for p in reports],
                                         "protocol": reference, "runs": runs, "summary": summary})
    write_json(output / "summary.json", summary)
    validation_profiles = {row["comparison_group"]: row.get("validation_profiles") for row in runs}
    # A learning curve can compare runs only when its validation target matches.
    common_validation = len({json.dumps(value) for value in validation_profiles.values()}) == 1
    plot_results(runs, summary, output, group_key="comparison_group", learning_curves=common_validation)
    write_json(output / "validation_profiles.json", validation_profiles)
    # Reuse the same plotting code for each train==test comparison. Cross-profile
    # scores stay in the root report and transfer matrices.
    for profile in dict.fromkeys(row["training_environment"]["profile"] for row in runs):
        if profile not in profiles:
            continue
        selected = [row for row in runs if row["training_environment"]["profile"] == profile]
        groups = dict.fromkeys(row["comparison_group"] for row in selected)
        matched_summary = {group: {**summary[group], "profiles": {profile: summary[group]["profiles"][profile]}}
                           for group in groups}
        directory = output / "by_training_profile" / profile
        directory.mkdir(parents=True, exist_ok=True)
        same_validation = len({json.dumps(row.get("validation_profiles")) for row in selected}) == 1
        plot_results(selected, matched_summary, directory, group_key="comparison_group",
                     learning_curves=same_validation)
    with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "test_profile", "n_train_seeds", "score_mean", "score_sd",
                         "reward_mean", "reward_sd", "success_10_mean", "success_50_mean",
                         "truncation_rate_mean", "training_seconds_mean"])
        for group, block in summary.items():
            for profile, metrics in block["profiles"].items():
                writer.writerow([group, profile, block["n_train_seeds"],
                                 metrics["score"]["mean"], metrics["score"]["std"],
                                 metrics["reward"]["mean"], metrics["reward"]["std"],
                                 metrics["success_rate_10"]["mean"], metrics["success_rate_50"]["mean"],
                                 metrics["truncation_rate"]["mean"],
                                 block["training_seconds"]["mean"]])
    with (output / "per_seed.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "seed", "actual_steps", "selected_steps", "training_seconds",
                         "first_validation_score_10_steps", "first_validation_score_10_seconds",
                         "first_validation_score_50_steps", "first_validation_score_50_seconds"])
        for row in runs:
            milestones = row["first_validation_to_score"]
            writer.writerow([row["comparison_group"], row["train_seed"], row["actual_train_steps"],
                             row["selected_steps"], row["cost"]["training_seconds"],
                             *[milestones[str(t)][key] if milestones[str(t)] else ""
                               for t in (10, 50) for key in ("steps", "training_seconds")]])
    # One row per learner/training setup, columns are test environments. All
    # values come from the existing between-seed summary, not new statistics.
    for statistic in ("mean", "std"):
        with (output / f"transfer_score_{statistic}.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["group", "method", "train_strategy", "train_profile", "n_train_seeds", *profiles])
            for group, block in summary.items():
                row = next(row for row in runs if row["comparison_group"] == group)
                setup = row["training_environment"]
                writer.writerow([group, row["method"], setup["strategy"], setup["profile"], block["n_train_seeds"],
                                 *[block["profiles"][profile]["score"][statistic] for profile in profiles]])
    print(f"Collected {len(runs)} runs: {output.resolve()}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--collect", nargs="+", type=Path, help="Completed/partial suite directories")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print all commands; do not train")
    parser.add_argument("--smoke", action="store_true", help="32-step integration check; not a scientific result")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--train-steps", type=int)
    parser.add_argument("--experiments", nargs="+", help="Run only named experiment ids")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/suites"))
    args = parser.parse_args(argv)
    if args.collect and any((args.dry_run, args.smoke, args.device is not None,
                             args.seeds is not None, args.train_steps is not None, args.experiments)):
        parser.error("--collect only accepts --output-dir; it does not train or change protocols")
    if args.collect:
        reports = []
        for directory in args.collect:
            manifest = json.loads((directory / "suite.json").read_text(encoding="utf-8"))
            reports.extend(Path(row["report"]) for row in manifest["experiments"] if row.get("report"))
        stamp = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).strftime("%Y%m%d_%H%M%S_%f")
        collect_reports(reports, args.output_dir / f"collected_{stamp}")
        return 0
    config = json.loads(args.config.read_text(encoding="utf-8"))
    plan = build_plan(config, smoke=args.smoke, device=args.device, seeds=args.seeds,
                      train_steps=args.train_steps, experiments=args.experiments)
    print(config["description"])
    print(f"{len(plan)} experiments, {sum(row['runs'] for row in plan)} training runs"
          + (" [SMOKE ONLY]" if args.smoke else ""))
    for row in plan:
        print(f"{row['id']} ({row['runs']} runs): python benchmark.py {shlex.join(row['arguments'])}")
    if args.dry_run:
        return 0
    from src.benchmarking import write_json
    stamp = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).strftime("%Y%m%d_%H%M%S_%f")
    output = args.output_dir.resolve() / (("smoke_" if args.smoke else "") + stamp)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"config": str(args.config.resolve()), "description": config["description"],
                "smoke_only": args.smoke, "status": "running", "experiments": plan}
    write_json(output / "suite.json", manifest)
    print(f"Suite directory: {output}")
    try:
        for row in plan:
            row["status"] = "running"
            write_json(output / "suite.json", manifest)
            experiment_dir = output / row["id"]
            benchmark.main([*row["arguments"], "--output-dir", str(experiment_dir)])
            row["report"] = str(next(experiment_dir.glob("*/results.json")).resolve())
            row["status"] = "complete"
            write_json(output / "suite.json", manifest)
        collect_reports([Path(row["report"]) for row in plan], output / "comparison")
        manifest["status"] = "complete"
    except BaseException as error:
        manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        raise
    finally:
        write_json(output / "suite.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
