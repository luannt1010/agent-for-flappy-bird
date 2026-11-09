"""Named physics and training setups shared by training and evaluation.

Algorithms stay in agent_registry; this module only configures the existing
BenchmarkEnv, DomainRandomization and Curriculum implementations.
"""

from dataclasses import asdict, replace

from src.benchmark_env import BenchmarkEnv, PhysicsConfig, PipeLayout
from src.curriculum import Curriculum
from src.domain_randomization import DomainRandomization


PROFILES = ("standard", "narrow", "fast", "wind", "combined", "surprise")
TRAIN_STRATEGIES = ("fixed", "domain_randomization", "curriculum")


def profile_physics(name: str) -> PhysicsConfig:
    """Return declared physics, in pixels and game steps; never tune on test."""
    changes = {
        "standard": {},
        "narrow": {"pipe_gap": 80.0},
        "fast": {"pipe_speed": -6.0},
        "wind": {"wind": 0.35},
        "combined": {"pipe_gap": 80.0, "pipe_speed": -6.0, "wind": 0.35},
        "surprise": {},
    }
    if name not in changes:
        raise ValueError(f"Unknown physics profile: {name}")
    return replace(PhysicsConfig(), **changes[name])


def profile_layout(name: str) -> PipeLayout:
    """Surprise changes gap placement; its clearance and physics stay standard."""
    if name not in PROFILES:
        raise ValueError(f"Unknown layout profile: {name}")
    return (PipeLayout(gap_top_min=80.0, gap_top_max=260.0, min_shift=80.0, max_shift=120.0)
            if name == "surprise" else PipeLayout())


def training_setup(method: str, profile="standard", strategy=None, curriculum_fraction=1.0) -> dict:
    """Resolve independent environment settings, preserving legacy method names.

    For curriculum, profile is the final endpoint. Randomization uses the
    existing absolute ranges, so only the standard base is accepted. Its
    ranges are recorded explicitly rather than implying a narrow-only run.
    """
    legacy = method if method in ("domain_randomization", "curriculum") else None
    if legacy and strategy not in (None, legacy):
        raise ValueError(f"Method {method} requires --train-strategy {legacy}")
    strategy = strategy or legacy or "fixed"
    if strategy not in TRAIN_STRATEGIES:
        raise ValueError(f"Unknown training strategy: {strategy}")
    base = profile_physics(profile)
    if method == "multi_agent" and (profile != "standard" or strategy != "fixed"):
        raise ValueError("multi_agent currently supports standard/fixed training only")
    if strategy == "domain_randomization" and profile != "standard":
        raise ValueError("domain_randomization uses absolute ranges; use --train-profile standard")
    if profile == "surprise" and strategy != "fixed":
        raise ValueError("surprise currently supports fixed training; compare placement separately from physics schedules")
    return {
        "profile": profile, "strategy": strategy, "physics": asdict(base),
        "randomization": asdict(DomainRandomization()) if strategy == "domain_randomization" else None,
        "curriculum": asdict(Curriculum(hard=base)) if strategy == "curriculum" else None,
        "curriculum_fraction": curriculum_fraction if strategy == "curriculum" else None,
        **({"layout": asdict(profile_layout(profile))} if profile == "surprise" else {}),
    }


def make_training_env(setup: dict, *, seed: int, max_episode_steps: int, render=False):
    """Build a single-player environment using the existing physics modifiers."""
    base = PhysicsConfig(**setup["physics"])
    schedule = setup["curriculum"]
    return BenchmarkEnv(
        seed=seed, max_episode_steps=max_episode_steps, render=render, physics=base,
        layout=PipeLayout(**setup["layout"]) if "layout" in setup else None,
        randomization=DomainRandomization(**setup["randomization"]) if setup["randomization"] else None,
        curriculum=Curriculum(easy=PhysicsConfig(**schedule["easy"]),
                              hard=PhysicsConfig(**schedule["hard"])) if schedule else None,
    )
