Project Overview: NEAT Game

This project involves a game where agents evolve their behavior using the NeuroEvolution of Augmenting Topologies (NEAT) algorithm. The focus is on clean, functional design and efficient headless simulation for training.
Technical Architecture
NEAT Implementation

    Topologies: The system must support dynamic growth of neural networks, including adding nodes and connections (crossover and mutation).

    Speciation: Group agents into species based on topological similarity to protect innovation.

    Fitness Function: Define clear metrics for agent success to guide evolution.

Simulation & Performance

    Headless Mode: Use the -simulate flag to run training cycles without rendering a GUI. This is the primary method for high-speed evolution.

    Decoupled Logic: Keep the physics and evolution logic separate from the rendering engine to ensure the simulation remains deterministic regardless of frame rate.

Design System
Visual Style

    Flat Design: Use solid colors and simple geometric shapes.

    No Post-Processing: Avoid glows, particles, or complex shaders.

    Clarity: Prioritize high contrast to ensure agent behavior is easily observable during GUI playback.