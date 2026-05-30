# Challenge 4: GAIL on ALE/Solaris-v5

## Quick Start

Run the comparison notebook to see all results:

```bash
jupyter nbconvert --to notebook --inplace --execute comparison_dqn_ppo_gail.ipynb
```

## Execution Commands

### 1. Collect Demonstrations from Best PPO
```bash
python SolarisGAIL.py --mode collect-demos \
    --checkpoint-path ppo_best/ppo_solaris_replicate_1.pth \
    --env-id "ALE/Solaris-v5" \
    --n-steps 50000 \
    --output-path demonstrations/demos_best_ppo.npz \
    --seed 42
```



### 2. Train GAIL from Best PPO Demos
```bash
python SolarisGAIL.py --mode train-gail \
    --demos demonstrations/demos_best_ppo.npz \
    --experiment gail_from_best_ppo \
    --run-seeds 3
```

### 3. Train GAIL from Mid PPO Demos
```bash
python SolarisGAIL.py --mode train-gail \
    --demos demonstrations/demos_best_ppo.npz \
    --experiment gail_from_mid_ppo \
    --run-seeds 3
```

## Logging & Artifacts

### Time Series Data
```
time_series/
├── dqn/rew_mean/          [DQN rewards]
├── ppo/mean_return/       [PPO learning curves]
└── gail/                  [GAIL curves - best & mid variants]
    ├── gail_from_best_ppo_best_seed/
    │   ├── mean_return_10ep.csv
    │   ├── disc_acc.csv
    │   ├── disc_loss.csv
    │   └── ...
    └── gail_from_mid_ppo_best_seed/
        ├── mean_return_10ep.csv
        ├── disc_acc.csv
        └── ...
```

### Model Checkpoints
```
models/
├── ppo_solaris_replicate_1.pth    [Seed 66064 - used for demos]
└── gail_solaris*.pth               [Trained GAIL models]
```

### TensorBoard Logs
```
logs/gail_solaris/
├── gail_from_best_ppo_seed_*/     [Best approach logs]
└── gail_from_mid_ppo_seed_*/      [Ablation logs]
```

Monitor with:
```bash
tensorboard --logdir logs/gail_solaris --port 6006
```

## Results Summary

| Algorithm | Final Return | Training Stability |
|-----------|--------------|-------------------|
| PPO       | 4764.00      | AUC = 0.451       |
| GAIL Best | 2064.00      | AUC = 0.313       |
| DQN       | 1365.20      | AUC = 0.563       |

## Generated Plots

- `learning_curves_comparison.png` - All three algorithms
- `gail_approaches_comparison.png` - Best vs mid demos
- `gail_discriminator_dynamics.png` - Discriminator acc & loss
- `final_performance_comparison.png` - Final returns

All in root directory after running notebook.


