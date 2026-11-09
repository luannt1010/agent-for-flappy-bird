"""Lazy algorithm registry used by the CLI and downstream experiments."""

from src.rl_common import AgentConfig


METHODS = (
    "dqn", "double_dqn", "dueling_dqn", "per_dqn", "rainbow",
    "ppo", "a2c", "sac", "domain_randomization", "curriculum", "multi_agent",
)


def make_agent(method: str, config: AgentConfig, device="cpu", seed: int = 0):
    """Create an agent from a registry name; return the common Agent interface.

    Environment extensions use the same DQN learner as the baseline, so their
    differences measure the training environment rather than a second algorithm.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}")
    if method == "rainbow":
        from src.rainbow import RainbowAgent
        return RainbowAgent(config, device=device, seed=seed)
    if method in ("ppo", "a2c", "sac"):
        if method == "ppo":
            from src.ppo import PPOAgent as AgentClass
        elif method == "a2c":
            from src.a2c import A2CAgent as AgentClass
        else:
            from src.sac import SACAgent as AgentClass
        return AgentClass(config, device=device, seed=seed)
    from src.dqn_agent import DQNAgent
    variant = method if method in ("dqn", "double_dqn", "dueling_dqn", "per_dqn") else "dqn"
    return DQNAgent(config, device=device, seed=seed, variant=variant)
