# [PYTORCH] Deep Q-learning for playing Flappy Bird

## Game demo

Run from the project root with the dependencies in `requirements-benchmark.txt`:

```bash
python game.py
python game.py --profile surprise
```

Without a checkpoint, play by keyboard. **Space / Up** flaps, **Enter** starts
or advances to the next game, **P** pauses, **R** retries the same seeded course,
and **Esc** exits. The window shows score, session best, reward and simulated
survival time. Completed rounds and session statistics are printed in the CLI;
restarting or quitting an unfinished round excludes it from those statistics.
The default speed is **30 FPS**. Use `--scale` to change window size.

Supply a single-agent benchmark `best.pt` to let an AI play. The algorithm and
training metadata are read from the checkpoint; no learning occurs during play:

```bash
python game.py --profile surprise --checkpoint benchmark_results2/compact/surprise/20261109_100919_805979/dqn/seed_0/best.pt
```

AI starts automatically and advances to the next course seed after each round.
Use `--episodes 3` to stop after three completed rounds, `--device cuda` for GPU
inference, or `--headless --episodes 3` to run without a window or FPS limit.
Other available profiles are `narrow`, `fast`, `wind` and `combined`.
The game uses the existing benchmark environment and checkpoint loader, with
the same collision rules, pipe score, preprocessing and episode time limit.
Demo session statistics describe the rounds played, not a replacement benchmark.
Run `python game.py --help` for all options.

## Extension benchmark

For the full four-learner × five-training-environment matrix, commands and outputs,
see [BENCHMARK_PLAN.md](BENCHMARK_PLAN.md). Preview without training:
`python benchmark_suite.py --config experiments/full_matrix.json --dry-run`.

See [BENCHMARK.md](BENCHMARK.md) for the modular DQN/Double/Dueling/PER/Rainbow,
PPO/A2C/discrete SAC, domain randomization, curriculum and competitive multi-agent
implementations. Run `python benchmark.py --help` for the CLI. Results, checkpoints,
JSON summaries and comparison plots are written to `benchmark_results/`.

## Introduction

Here is my python source code for training an agent to play flappy bird. It could be seen as a very basic example of Reinforcement Learning's application.
<p align="center">
  <img src="demo/flappybird.gif" width=600><br/>
  <i>Result</i>
</p>

## How to use my code

With my code, you can:
* **Train your model from scratch** by running **python train.py**
* **Test your trained model** by running **python test.py**

## Trained models

You could find my trained model at **trained_models/flappy_bird**
 
## Requirements

* **python 3.6**
* **pygame**
* **cv2**
* **pytorch** 
* **numpy**
