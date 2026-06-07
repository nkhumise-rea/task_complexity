# Issues with Measuring Task Complexity via Random Policies in Robotic Tasks

[![Published at AAMAS 2026](https://img.shields.io/badge/AAMAS-2026-blue.svg)](https://dl.acm.org/doi/10.65109/FDIK3367)
[![Paper (PDF)](https://img.shields.io/badge/paper-PDF-blue.svg)](https://dl.acm.org/doi/10.65109/FDIK3367)

Reference implementation for the paper **"Issues with Measuring Task Complexity via Random Policies in Robotic Tasks"** (Nkhumise, Talamali & Gilra), published in *Proceddings of the 25th International Conference on Autonomous Agents and Multiagent Systems* (AAMAS), 2026 — [paper](https://dl.acm.org/doi/10.65109/FDIK3367). For detailed derivations and extended results on the paper, see [supplementary_material.pdf](https://github.com/nkhumise-rea/task_complexity/blob/main/Supplementary_material.pdf).

## Background and Motivation

Reinforcement learning (RL) has enabled major advances in fields such as robotics and natural language processing. A key challenge in RL is measuring task complexity, which is essential for creating meaningful benchmarks and designing effective curricula. While there are numerous well-established metrics for assessing task complexity in tabular settings, relatively few exist in non-tabular domains. These include (i) Statistical analysis of the performance of random policies via **Random Weight Guessing (RWG)**, and (ii) information-theoretic metrics **Policy Information Capacity (PIC)** and **Policy-Optimal Information Capacity (POIC)**, which are reliant on RWG. 

In this work, we evaluate these methods using progressively difficult robotic manipulation setups, with known relative complexity, with both dense and sparse reward formulations. Our empirical results reveal that measuring complexity is still nuanced. Specifically, under the same reward formulation, PIC suggests that a two-link robotic arm setup is easier than a single-link setup --- which contradicts the robotic control and empirical RL perspective whereby the two-link setup is inherently more complex. Likewise, for the same setup, POIC estimates that tasks with sparse rewards are easier than those with dense rewards. Thus, we show that both PIC and POIC contradict typical understanding and empirical results from RL. These findings highlight the need to move beyond RWG-based metrics towards better metrics that can more reliably capture task complexity in non-tabular RL with our task framework as a starting point.

## Repository Structure

```
task_complexity/
├── pic/                          # Core library (installable Python package)
│   ├── algos/
│   │   └── numpyagent.py         # NumPy-based policy agent (deterministic & stochastic)
│   ├── nn/
│   │   └── numpymlp.py           # Pure-NumPy MLP with configurable initialisation
│   ├── sampler/
│   │   ├── sampler.py            # Parallel episode sampler for reward collection
│   │   └── sampler_HER.py        # Sampler with Hindsight Experience Replay support
│   └── gym/
│       ├── reward_shaping/       # Reward shaping wrappers (MuJoCo, maze, reacher)
│       ├── noisy_dynamics/       # Noisy-dynamics wrappers (CartPole, HalfCheetah, Humanoid)
│       └── multi_step/           # Multi-step environment wrapper
│
├── examples/
│   ├── tasks/
│   │   ├── single_arm/           # 1-DOF arm reaching task (PyBullet)
│   │   │   ├── task.py           # Environment: 9D state, 1D torque action
│   │   │   ├── rwg.py            # Random Weight Generation data collection
│   │   │   ├── arm.urdf          # URDF robot model
│   │   │   └── ...               # Ablation variants (torque limits, initialisations)
│   │   └── double_arm/           # 2-DOF arm reaching task (PyBullet)
│   │       ├── task.py           # Environment: 10D state, 2D torque action
│   │       ├── task_obstacle.py  # Extended variant with collision avoidance
│   │       ├── kinematics.py     # Forward/inverse kinematics for 2-link arm
│   │       ├── rwg.py            # Random Weight Generation data collection
│   │       └── ...               # HER variants, obstacle variants, datasets (.npy)
│   └── data_results/
│       ├── mi_estimate.py        # PIC & POIC computation (Optuna-optimised temperature)
│       ├── mi_estimate_batch.py  # Batch MI estimation across configurations
│       ├── bootstrapping.py      # Bootstrap confidence intervals (1000 iterations)
│       ├── WelchTtest.py         # Welch's t-test for pairwise task comparisons (PIC)
│       ├── WelchTtest_poic.py    # Welch's t-test for pairwise comparisons (POIC)
│       ├── Plots_of_measure.py   # Correlation plots: metrics vs. RL performance
│       ├── display_dist.py       # Distribution visualisation
│       └── batches/              # Batch experiment outputs (metrics & tables)
│
├── RL_agents/                    # Reinforcement learning baselines
│   ├── single_arm/
│   │   ├── envs/                 # RL-specific environment variants
│   │   │   ├── task.py           # Single-arm env for SAC/DDPG training
│   │   │   └── task_HER.py       # HER-compatible variant
│   │   ├── tianshou/             # Tianshou-based training scripts
│   │   │   ├── sac_test.py       # SAC training
│   │   │   ├── ddpg_test.py      # DDPG training
│   │   │   ├── *_HER.py          # HER-augmented variants
│   │   │   └── eval_trained_*.py # Evaluation of trained policies
│   │   └── default_scripts/      # Baseline RL scripts and saved models
│   └── double_arm/
│       ├── envs/
│       │   ├── task_db.py        # Double-arm env for RL training
│       │   ├── task_obstacle.py  # Obstacle variant for RL
│       │   ├── kinematics.py     # IK solver for the 2-link arm
│       │   └── eq_motion/        # Equations of motion & dynamics
│       └── tianshou/             # Training & evaluation scripts
│
├── sac_trained/                  # SAC training results & visualisation
│   ├── dense_settings.py         # Learning curve plots (dense reward tasks)
│   ├── sparse_settings.py        # Learning curve plots (sparse reward tasks)
│   ├── training_plots.py         # General training visualisation
│   └── *.csv, *.png              # Training data and generated figures
│
├── setup.py                      # Package installation (pip install -e .)
├── environment_hpc.yml           # Conda environment for HPC reproduction
└── Task Complexity Paper Results.xlsx  # Consolidated paper results
```

## Task Environments

Four robotic reaching tasks of increasing complexity, all implemented in PyBullet:

| Task | DOF | State Dim | Action Dim | Description |
|------|-----|-----------|------------|-------------|
| **1-Link Dense (100)** | 1 | 9 | 1 | Single-arm reaching, target at fixed radius (~1.0m), dense reward |
| **1-Link Dense (170)** | 1 | 9 | 1 | Single-arm reaching, target at extended radius (~1.65m), dense reward |
| **2-Link Dense** | 2 | 10 | 2 | Double-arm reaching, variable radius (0.35–1.51m), dense reward |
| **2-Link Dense + Obstacle** | 2 | 10 | 2 | Double-arm reaching with collision avoidance, dense reward |

Each task also has a **sparse reward** variant where the agent receives reward only within a distance threshold of the goal.

**Dense reward**: `r = -(distance² + torque²)`
**Sparse reward**: `r = 1 if distance < 0.05 [meters] else 0`
**Obstacle penalty**: additional terms for collision (`-1000`) and proximity to obstacle

## Method

### Random Weight Generation (RWG)

The core data collection procedure:
1. Sample random neural network weights from a configurable distribution (normal, uniform, Xavier)
2. Execute episodes in the task environment using the random policy
3. Record the cumulative reward distribution across weight samples
4. Repeat for multiple independent samples (typically 200 samples x 500 episodes)

Reference: [`Oller et. al (2020)`](https://arxiv.org/abs/2004.07707)

### Information-Theoretic Measures

From the collected reward distributions:
- **PIC (Policy-Reward Information Content)**: Mutual information between network weights and episodic cumulative rewards — measures the interdependency between policy parameters (weights) and task performance.
- **POIC (Policy-Optimality Information Content)**: Mutual information between network weights and a binary optimality indicator — measures the interdependency between policy parameters and optimal behaviour.

Reference: [`Furuta et. al (2021)`](https://arxiv.org/abs/2103.12726)

### Statistical Analysis

- **Bootstrap resampling** (1000 iterations) for confidence intervals on PIC and POIC estimates
- **Welch's t-test** for pairwise significance testing between task configurations
- **Pearson correlation** between information measures and RL agent performance

## RL Baselines

Reinforcement learning agents trained via [Tianshou](https://github.com/thu-ml/tianshou) to validate the complexity ordering:
- **SAC** (Soft Actor-Critic)
- **SAC + HER** (with Hindsight Experience Replay for sparse reward tasks)

Training curves are normalised and compared across task variants in `sac_trained/`.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd task_complexity

# Create conda environment (for HPC / full reproduction)
conda env create -f environment_hpc.yml
conda activate pic_env

# Or install the core package directly
pip install -e .
```

### Requirements

- Python 3.8+
- PyTorch >= 1.5.1
- PyBullet (via `gym`)
- NumPy, SciPy, Matplotlib, Seaborn, Pandas
- Optuna (for temperature optimisation)
- Tianshou (for RL baselines)

## Usage

### Data Collection (Random Weight Generation)

```bash
# Single-arm task
python examples/tasks/single_arm/rwg.py

# Double-arm task
python examples/tasks/double_arm/rwg.py

# Double-arm with obstacle
python examples/tasks/double_arm/rwg_obstacle.py
```

### Computing PIC & POIC

```bash
python examples/data_results/mi_estimate.py
```

### Statistical Tests

```bash
# Bootstrap confidence intervals
python examples/data_results/bootstrapping.py

# Welch's t-test across tasks
python examples/data_results/WelchTtest.py
```

### Training RL Baselines

```bash
# SAC on single-arm
python RL_agents/single_arm/tianshou/sac_test.py

# DDPG on double-arm
python RL_agents/double_arm/tianshou/ddpg_test.py
```

### HPC Execution

Shell scripts (`zope.sh`, `zope_rev.sh`) are provided for job submission on HPC clusters.

## Supplementary Material

The file [`supplementary_material.pdf`](https://github.com/nkhumise-rea/task_complexity/blob/main/Supplementary_material.pdf) contains:
- Specification of employed SAC hyperparameters 
- Bootstrapping and statistical significance testing results
- Extended experimental results 
- Detailed description of the **double-arm + obstacle** task configuration

Consolidated numerical results from all experiments are available in [`Task Complexity Paper Results.xlsx`](https://github.com/nkhumise-rea/task_complexity/blob/main/Task%20Complexity%20Paper%20Results.xlsx).

## Citation

```bibtex
@article{nkhumise2026,
  title   = {Issues with Measuring Task Complexity via Random Policies in Robotic Tasks},
  author  = {Nkhumise, Reabetswe M. and Talamali, Mohamed S. and Gilra, Aditya},
  journal = {Proc. of the 25th International Conference on Autonomous Agents and Multiagent Systems (AAMAS)},
  year    = {2026},
  url     = {https://dl.acm.org/doi/10.65109/FDIK3367}
}
```
