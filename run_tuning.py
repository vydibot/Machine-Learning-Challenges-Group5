#!/usr/bin/env python3
"""
Run Optuna hyperparameter tuning for Solaris DQN agent.
"""

from Solaris import SolarisHyperparameterTuner
import optuna

# Example: Use TPESampler (Bayesian optimization)
tuner = SolarisHyperparameterTuner(sampler=optuna.samplers.TPESampler())

# Run optimization with 10 trials (increase for better results)
tuner.optimize(n_trials=10)

# Save the best configuration to sweep_configs.json
tuner.save_to_sweep_config()

print("Optimization complete. Best config saved to sweep_configs.json")