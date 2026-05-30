# Checklist

## Exact command to reproduce the best PPO run

Use the final chosen PPO replicate run:

```bash
python Solaris.py --mode play --model-path models/ppo_solaris_replicate_1 --episodes 3
```

## Seeds used for PPO repeated experiments

- `ppo_solaris_gae_variance_reduction (Final choose model) 3 seeds expirment_replicate_1`: `52372`
- `ppo_solaris_gae_variance_reduction (Final choose model) 3 seeds expirment_replicate_2`: `66064`
- `ppo_solaris_gae_variance_reduction (Final choose model) 3 seeds expirment_replicate_3`: `95921`

## Pointers to logs and figures

- PPO logs: `logs/ppo_solaris/replicates/` and `logs/ppo_solaris/sweep/`
- PPO time-series: `time_series/ppo/mean_return/`, `time_series/ppo/policy_loss/`, `time_series/ppo/value_loss/`
- DQN time-series: `time_series/dqn/rew_mean/`, `time_series/dqn/loss/`, `time_series/dqn/len_mean/`
- Comparison and figure outputs: `variance_analysis/ppo_vs_dqn_comparison.ipynb`, `variance_analysis/ppo_vs_dqn_improved.ipynb`, `variance_analysis/ppo_vs_dqn_summary.csv`, `variance_analysis/figures/`

## Empirical algorithmic difference observed

In the Solaris environment, the empirical comparison shows PPO as a higher-reward, higher-variance method while DQN behaves as a more stable optimizer that typically converges to a local maximum. PPO’s on-policy updates and clipped surrogate objective allow it to adapt policy parameters aggressively from fresh trajectories, leading to more chaotic learning traces but ultimately higher returns in the best runs. DQN maintains a more conservative value-based update style with experience replay, so its reward progress is smoother and its performance variance is lower. As a result, DQN often learns a reliable policy quickly and then improves only modestly after finding a good action-value estimate, making it appear stable but limited by a local optimum. PPO, by contrast, continues exploring alternate behaviors and can break through that plateau at the expense of less predictable episode-to-episode outcomes. This matches the observed data: DQN is robust and low variance, while PPO is less stable yet capable of reaching superior final performance on Solaris. In practice, this means DQN produces consistent policies, whereas PPO’s greater exploration and variance let it achieve superior final performance on Solaris. The empirical pattern supports selecting DQN for steadiness and PPO for maximum reward despite its chaotic training behavior.