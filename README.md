# Cat Qubit PPO Calibration

## Purpose

- Trains a PPO agent to tune cat-qubit stabilization knobs.
- Uses `dynamiqs` for quantum simulation and `PyTorch` for PPO.
- Optimizes for:
  - target bias: `eta = Tz / Tx`
  - large logical lifetimes: `Tz` and `Tx`
  - stable photon occupations within numerical cutoffs

## Main File

- `solution.ipynb`

## Inputs

- No external dataset required.
- Main configuration is defined in `CatConfig`:
  - `na`, `nb`: Hilbert-space cutoffs
  - `kappa_b`, `kappa_a`: fixed loss rates
  - `default_knobs`: starting values for `[g2, eps_d]`
  - `low`, `high`: knob bounds
  - `target_eta`: target logical bias
  - `drift_std`: simulated hardware drift strength
  - `episode_len`: PPO episode length
- PPO settings are defined in `PPOConfig`:
  - `total_updates`
  - `rollout_steps`
  - `ppo_epochs`
  - `minibatch_size`
  - `lr`
  - `gamma`
  - `gae_lambda`

## Tuned Parameters

- The optimizer tunes two physical knobs:
  - `g2`
  - `eps_d`

## Outputs

- Console logs for each PPO update:
  - update number
  - global step
  - mean reward
  - latest `Tz`, `Tx`, and `eta`
  - best reward found
  - best knob values
- Saved files:
  - `training_history.json`
  - `training_history.png`

## How It Works

- Builds a two-mode cat-qubit Hamiltonian.
- Simulates stabilization from vacuum.
- Discovers the cat manifold using the Husimi-Q function.
- Estimates:
  - `Tz` from logical Z decay
  - `Tx` from logical X decay
- Computes reward from:
  - lifetime score
  - bias error
  - cutoff and buffer penalties
- PPO learns knob updates under simulated drift.

## Running

- Open `solution.ipynb`.
- Run all cells from top to bottom.
- The final cell starts training and saves the output files.

## Key Dependencies

- `numpy`
- `scipy`
- `torch`
- `dynamiqs`
- `jax.numpy`
- `matplotlib`

## Important Notes

- `kappa_a` and `kappa_b` are fixed during PPO training.
- Hardware drift is simulated with a random walk.
- The current environment exposes drift in the observation.
- The lifetime estimates use exponential curve fitting.
- Failed simulations receive a large negative reward.
