# NEAT Games: Flappy Bird and Inverted Pendulum

Two small demos that evolve agents with NEAT:
- A Flappy Bird clone where a neural net learns to flap through pipes.
- An inverted pendulum simulation where a cart keeps the pole balanced.

## Repo Layout
- [FlappyBirdAi](FlappyBirdAi): Pygame Flappy Bird, NEAT setup, sprites, audio.
- [InvertedPendulumAi](InvertedPendulumAi): Pygame inverted pendulum, NEAT setup.
- [utils/visualize.py](utils/visualize.py): Helpers to plot fitness curves and render networks.
- [requirements.txt](requirements.txt): Python dependencies (pygame, neat-python, matplotlib, graphviz bindings, etc.).

## Prerequisites
- Python 3.10+ recommended.
- System Graphviz binaries for network rendering (install from https://graphviz.org/download/).
- On Windows, `SDL_VIDEODRIVER` issues are rare, but ensure a desktop session is available when running pygame.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the Flappy Bird experiment
```bash
python FlappyBirdAi/game/flappy.py
```
- Uses the NEAT config at [FlappyBirdAi/config-flappybird.txt](FlappyBirdAi/config-flappybird.txt) by default. Override with env var `FLAPPY_CONFIG` if you want to point to another config file.
- Assets and sounds live in [FlappyBirdAi/game/assets](FlappyBirdAi/game/assets).
- The script loads env vars from a local `.env` (optional) via `python-dotenv`.
- Training runs for 200 generations in the main block and visualizes the best network when finished.

## Run the Inverted Pendulum experiment
```bash
python InvertedPendulumAi/game/pendulum.py
```
- Uses the NEAT config at [InvertedPendulumAi/config-inverted-pendulum.txt](InvertedPendulumAi/config-inverted-pendulum.txt).
- Fitness is primarily survival time; networks pick a cart action (`Left`, `None`, `Right`) each frame.
- Outputs (winner network graph and fitness plots) are written under `InvertedPendulumAi/output` if the directory exists; a fallback call also renders to the default Graphviz location.

## Visualizing results
- Fitness and species curves: `plot_stats` and `plot_species` in [utils/visualize.py](utils/visualize.py).
- Network graphs: `draw_net` in [utils/visualize.py](utils/visualize.py) (requires Graphviz installed and on PATH).

## Tips
- If pygame cannot open a window, ensure you are not running headless or over SSH without a display.
- For quicker tests, lower generation counts in the `p.run(...)` calls inside each game script.


