"""Interactive Flappy Bird demo using the benchmark's physics and scoring.

Human: python game.py --profile surprise
AI:    python game.py --profile surprise --checkpoint path/to/best.pt

SPACE/UP: flap or start the next game. ENTER: start/next game.
P: pause/resume. R: retry the same course. ESC: quit.
AI starts automatically and advances to the next seed after each game.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame
import torch

from src.benchmark_env import BenchmarkEnv
from src.benchmarking import load_checkpoint, summarize_episodes
from src.experiment_config import PROFILES, profile_layout, profile_physics


def get_args(argv=None):
    """Validate CLI options; a checkpoint selects AI instead of keyboard play."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", choices=PROFILES, default="standard")
    parser.add_argument("--checkpoint", type=Path, help="Single-agent benchmark best.pt; omit to play by keyboard")
    parser.add_argument("--device", default="cpu", help="cpu, cuda or cuda:N; used for AI inference")
    parser.add_argument("--seed", type=int, default=2_000_000, help="First course seed; next game increments it")
    parser.add_argument("--fps", type=int, default=30, help="30 = normal speed; 0 = uncapped")
    parser.add_argument("--scale", type=float, default=1.5, help="Display scale, reduced automatically to fit the screen")
    parser.add_argument("--episodes", type=int, default=0, help="Stop after this many completed games; 0 = interactive session")
    parser.add_argument("--max-episode-steps", type=int, default=10_000)
    parser.add_argument("--headless", action="store_true", help="AI only: run without a window or FPS limit; defaults to 1 episode")
    args = parser.parse_args(argv)
    if args.seed < 0 or args.fps < 0 or args.episodes < 0:
        parser.error("--seed, --fps and --episodes must be nonnegative")
    if args.max_episode_steps < 1:
        parser.error("--max-episode-steps must be positive")
    if not math.isfinite(args.scale) or args.scale <= 0:
        parser.error("--scale must be finite and positive")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    if args.headless and args.checkpoint is None:
        parser.error("--headless requires --checkpoint; keyboard play needs a window")
    if args.headless and args.episodes == 0:
        args.episodes = 1
    return args


class GameSession:
    """Manage demo rounds while delegating actions, collisions and score to the environment."""

    def __init__(self, profile, seed, max_episode_steps, agent=None):
        self.env = BenchmarkEnv(max_episode_steps=max_episode_steps,
                                physics=profile_physics(profile), layout=profile_layout(profile))
        self.agent = agent
        self.seed = seed
        self.best_score = 0
        self.episodes = []
        self.reset(seed)

    def reset(self, seed=None):
        """Retry the current seed or start another course; unfinished games are not recorded."""
        if seed is not None:
            self.seed = seed
        self.state = self.env.reset(seed=self.seed)
        self.total_reward = 0.0
        self.ended = False

    def step(self, action=0):
        """Advance once; AI always uses deterministic inference without learning."""
        if self.ended:
            raise RuntimeError("Reset the session before playing another game.")
        if self.agent is not None:
            action = self.agent.act(self.state, explore=False)
        self.state, reward, terminated, truncated, info = self.env.step(action)
        self.total_reward += float(reward)
        self.best_score = max(self.best_score, int(info["score"]))
        self.ended = terminated or truncated
        if self.ended:
            record = {key: info[key] for key in ("seed", "score", "episode_steps", "survival_seconds")}
            record.update(player_id=0, reward=self.total_reward,
                          terminated=bool(terminated), truncated=bool(truncated))
            self.episodes.append(record)
            return record
        return None

    def close(self):
        self.env.close()


class GameDisplay:
    """Draw the environment frame and a separate HUD; never draw into model inputs."""

    def __init__(self, scale):
        pygame.display.init()
        pygame.font.init()
        screen = pygame.display.Info()
        scale = min(scale, max(0.5, (screen.current_w - 48) / BenchmarkEnv.screen_width),
                    max(0.5, (screen.current_h - 170) / BenchmarkEnv.screen_height))
        self.game_size = (max(1, round(BenchmarkEnv.screen_width * scale)),
                          max(1, round(BenchmarkEnv.screen_height * scale)))
        width, height = self.game_size
        self.font = pygame.font.Font(None, max(16, round(width / 24)))
        self.title_font = pygame.font.Font(None, max(26, round(width / 13)))
        self.line_height = self.font.get_linesize()
        hud_height = self.title_font.get_linesize() + 4 * self.line_height + 20
        self.window = pygame.display.set_mode((width, height + hud_height))
        pygame.display.set_caption("Flappy Bird Demo")

    def draw(self, session, profile, controller, phase):
        width, height = self.game_size
        frame = pygame.transform.scale(session.env.render_surface(), self.game_size)
        self.window.fill((18, 23, 32))
        self.window.blit(frame, (0, 0))
        if phase != "running":
            overlay = pygame.Surface(self.game_size, pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160))
            self.window.blit(overlay, (0, 0))
            title = {"ready": "Ready", "paused": "Paused", "over": "Game over"}[phase]
            if phase == "over" and session.episodes[-1]["truncated"]:
                title = "Time limit reached"
            hint = "P: resume" if phase == "paused" else "SPACE / ENTER: play"
            if phase == "over" and session.agent is not None:
                hint = "Next game starts automatically"
            for text, font, y in [(title, self.title_font, height // 2 - 30),
                                  (hint, self.font, height // 2 + 10)]:
                rendered = font.render(text, True, (255, 255, 255))
                self.window.blit(rendered, rendered.get_rect(center=(width // 2, y)))

        score = self.title_font.render(f"Score: {session.env.score}   Best: {session.best_score}", True, (255, 210, 90))
        y = height + 8
        self.window.blit(score, (10, y))
        y += self.title_font.get_linesize()
        lines = [f"{controller} | {profile} | seed {session.seed}",
                 f"Reward: {session.total_reward:.1f} | Time: {session.env.episode_steps / session.env.fps:.1f}s",
                 "SPACE / UP: flap   P: pause",
                 "R: retry same course   ESC: quit"]
        for text in lines:
            self.window.blit(self.font.render(text, True, (230, 234, 240)), (10, y))
            y += self.line_height
        pygame.display.flip()

    def close(self):
        pygame.display.quit()
        pygame.font.quit()


def play(args, agent=None, controller="Human"):
    """Run keyboard/AI play; return only completed rounds for descriptive session statistics."""
    session = GameSession(args.profile, args.seed, args.max_episode_steps, agent)
    display = None
    try:
        display = None if args.headless else GameDisplay(args.scale)
        clock = pygame.time.Clock() if display is not None else None
        phase = "running" if agent is not None else "ready"
        finished_at = 0.0
        running = True
        with torch.inference_mode():
            while running:
                flap = False
                for event in pygame.event.get() if display is not None else ():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r:
                            session.reset()
                            phase = "ready"
                        elif event.key == pygame.K_p and phase in ("running", "paused"):
                            phase = "paused" if phase == "running" else "running"
                        elif event.key in (pygame.K_SPACE, pygame.K_UP, pygame.K_RETURN):
                            if phase == "over":
                                session.reset(session.seed + 1)
                            if phase in ("ready", "over"):
                                phase = "running"
                            flap = flap or event.key in (pygame.K_SPACE, pygame.K_UP)
                if not running:
                    break
                if phase == "over" and agent is not None and (args.headless or time.monotonic() - finished_at >= 1.2):
                    session.reset(session.seed + 1)
                    phase = "running"
                if phase == "running":
                    record = session.step(int(flap))
                    if record is not None:
                        reason = "time limit" if record["truncated"] else "collision"
                        print(f"Game {len(session.episodes)} | seed={session.seed} | score={record['score']} | "
                              f"reward={record['reward']:.2f} | steps={record['episode_steps']} | {reason}", flush=True)
                        phase, finished_at = "over", time.monotonic()
                        if args.episodes and len(session.episodes) >= args.episodes:
                            break
                if display is not None:
                    display.draw(session, args.profile, controller, phase)
                    clock.tick(args.fps if phase == "running" else 30)
    except KeyboardInterrupt:
        print("\nSession stopped.")
    finally:
        session.close()
        if display is not None:
            display.close()
        if session.episodes:
            summary = summarize_episodes(session.episodes)
            print(f"Completed: {summary['episodes']} | Best: {max(row['score'] for row in session.episodes)} | "
                  f"Mean score: {summary['score']['mean']:.2f} | Mean reward: {summary['reward']['mean']:.2f}")
    return session.episodes


def main(argv=None):
    args = get_args(argv)
    agent, controller = None, "Human"
    if args.checkpoint is not None:
        device = torch.device(args.device)
        if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
            raise ValueError("Choose cpu or an available CUDA device.")
        torch.set_num_threads(1)
        agents, metadata = load_checkpoint(args.checkpoint, str(device))
        if len(agents) != 1:
            raise ValueError("The demo expects a single-agent benchmark checkpoint.")
        agent = agents[0]
        controller = metadata["method"]
        train_profile = (metadata.get("training_environment") or {}).get("profile", "unknown")
        print(f"AI: {controller} | training seed={metadata['train_seed']} | "
              f"training profile={train_profile} | checkpoint step={metadata['steps']}")
    print(f"Profile: {args.profile} | Controller: {controller} | Initial seed: {args.seed}")
    if not args.headless:
        print("SPACE/UP: flap | ENTER: start/next | P: pause | R: retry same seed | ESC: quit")
    play(args, agent, controller)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, pygame.error) as error:
        raise SystemExit(f"Error: {error}") from error
