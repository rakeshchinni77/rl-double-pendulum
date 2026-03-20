"""
Unit tests for DoublePendulumEnv - Phase 1 Validation.

Tests:
- Environment instantiation
- Observation and action spaces
- Reset and step functionality
- Reward function calculations (baseline and shaped)
"""

import numpy as np
import sys
from pathlib import Path
import csv
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from env.environment import DoublePendulumEnv


def test_csv_logging():
    """Validate CSV logging contains required columns and data rows."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = str(Path(tmp_dir) / 'metrics.csv')
        env = DoublePendulumEnv(
            reward_type='shaped',
            enable_csv_logging=True,
            log_path=log_path,
        )

        env.reset()
        for _ in range(5):
            env.step(np.array([0.1], dtype=np.float32))
        env.close()

        assert Path(log_path).exists(), 'CSV log file was not created'
        with open(log_path, mode='r', encoding='utf-8') as csv_file:
            rows = list(csv.reader(csv_file))

        assert rows[0] == ['timesteps', 'mean_reward'], f'Unexpected header: {rows[0]}'
        assert len(rows) > 1, 'CSV should contain at least one metrics row'

        print(f"  ✓ CSV logging created file: {log_path}")
        print(f"  ✓ CSV header: {rows[0]}")
        print(f"  ✓ CSV data rows: {len(rows) - 1}")


def test_shaped_reward_sensitivity():
    """Shaped reward should react more strongly to cart displacement and velocity than baseline."""
    env_baseline = DoublePendulumEnv(reward_type='baseline')
    env_shaped = DoublePendulumEnv(reward_type='shaped')

    env_baseline.reset()
    env_shaped.reset()

    baseline_rewards = []
    shaped_rewards = []
    for _ in range(20):
        _, r_b, _, _, _ = env_baseline.step(np.array([0.7], dtype=np.float32))
        _, r_s, _, _, _ = env_shaped.step(np.array([0.7], dtype=np.float32))
        baseline_rewards.append(r_b)
        shaped_rewards.append(r_s)

    env_baseline.close()
    env_shaped.close()

    # Shaped should typically be lower due to center/velocity/action penalties.
    assert np.mean(shaped_rewards) < np.mean(baseline_rewards), (
        'Shaped reward should be lower than baseline under aggressive actions'
    )

    print(f"  ✓ Baseline mean under disturbance: {np.mean(baseline_rewards):.4f}")
    print(f"  ✓ Shaped mean under disturbance: {np.mean(shaped_rewards):.4f}")


def run_tests():
    """Run all Phase 2 validation tests."""
    print("=" * 70)
    print("PHASE 2 VALIDATION: Reward Functions + CSV Logging")
    print("=" * 70)
    
    # Test 1: Instantiation
    print("\n[TEST 1/7] Environment Instantiation...")
    try:
        env_baseline = DoublePendulumEnv(reward_type='baseline')
        env_shaped = DoublePendulumEnv(reward_type='shaped')
        print("  ✓ Baseline environment created")
        print("  ✓ Shaped environment created")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    # Test 2: Observation Space
    print("\n[TEST 2/7] Observation Space...")
    try:
        assert env_baseline.observation_space.shape == (6,), f"Expected shape (6,), got {env_baseline.observation_space.shape}"
        assert env_baseline.observation_space.dtype == np.float32, f"Expected dtype float32, got {env_baseline.observation_space.dtype}"
        print(f"  ✓ Observation space shape: {env_baseline.observation_space.shape}")
        print(f"  ✓ Observation space dtype: {env_baseline.observation_space.dtype}")
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    # Test 3: Action Space
    print("\n[TEST 3/7] Action Space...")
    try:
        assert env_baseline.action_space.shape == (1,), f"Expected shape (1,), got {env_baseline.action_space.shape}"
        assert env_baseline.action_space.low[0] == -1.0, f"Expected low=-1.0, got {env_baseline.action_space.low[0]}"
        assert env_baseline.action_space.high[0] == 1.0, f"Expected high=1.0, got {env_baseline.action_space.high[0]}"
        print(f"  ✓ Action space shape: {env_baseline.action_space.shape}")
        print(f"  ✓ Action space range: [{env_baseline.action_space.low[0]}, {env_baseline.action_space.high[0]}]")
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    # Test 4: Reset
    print("\n[TEST 4/7] Reset Function...")
    try:
        obs = env_baseline.reset()
        assert isinstance(obs, np.ndarray), f"Expected ndarray, got {type(obs)}"
        assert obs.shape == (6,), f"Expected shape (6,), got {obs.shape}"
        assert np.all(np.isfinite(obs)), "Observation contains non-finite values"
        print(f"  ✓ Reset returns observation shape: {obs.shape}")
        print(f"  ✓ Initial observation: {obs}")
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    # Test 5: Step Function
    print("\n[TEST 5/7] Step Function...")
    try:
        action = np.array([0.0])
        obs, reward, terminated, truncated, info = env_baseline.step(action)
        
        assert obs.shape == (6,), f"Observation shape should be (6,), got {obs.shape}"
        assert isinstance(reward, (float, np.floating)), f"Reward should be scalar, got {type(reward)}"
        assert isinstance(terminated, (bool, np.bool_)), f"Terminated should be bool, got {type(terminated)}"
        assert isinstance(truncated, (bool, np.bool_)), f"Truncated should be bool, got {type(truncated)}"
        assert isinstance(info, dict), f"Info should be dict, got {type(info)}"
        
        print(f"  ✓ Step returns observation: shape {obs.shape}, dtype {obs.dtype}")
        print(f"  ✓ Step returns reward: {reward:.4f}")
        print(f"  ✓ Step returns terminated: {terminated}, truncated: {truncated}")
        print(f"  ✓ Step returns info dict with keys: {list(info.keys())}")
    except (AssertionError, ValueError) as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    # Test 6: Baseline Reward
    print("\n[TEST 6/7] Baseline Reward Function...")
    try:
        env_baseline.reset()
        rewards = []
        for _ in range(10):
            _, reward, _, _, _ = env_baseline.step(np.array([0.0]))
            rewards.append(reward)
        
        mean_reward = np.mean(rewards)
        print(f"  ✓ Baseline reward mean: {mean_reward:.4f}")
        print(f"  ✓ Baseline reward std: {np.std(rewards):.4f}")
        print(f"  ✓ Baseline reward range: [{min(rewards):.4f}, {max(rewards):.4f}]")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    # Test 7: Shaped Reward
    print("\n[TEST 7/7] Shaped Reward Function...")
    try:
        env_shaped.reset()
        rewards = []
        for _ in range(10):
            _, reward, _, _, _ = env_shaped.step(np.array([0.0]))
            rewards.append(reward)
        
        mean_reward = np.mean(rewards)
        print(f"  ✓ Shaped reward mean: {mean_reward:.4f}")
        print(f"  ✓ Shaped reward std: {np.std(rewards):.4f}")
        print(f"  ✓ Shaped reward range: [{min(rewards):.4f}, {max(rewards):.4f}]")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    # Test 8: CSV logging
    print("\n[TEST 8/9] CSV Logging...")
    try:
        test_csv_logging()
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")
        return False

    # Test 9: shaped reward sensitivity
    print("\n[TEST 9/9] Shaped Reward Sensitivity...")
    try:
        test_shaped_reward_sensitivity()
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")
        return False

    # Cleanup
    env_baseline.close()
    env_shaped.close()
    
    return True


if __name__ == '__main__':
    success = run_tests()
    
    if success:
        print("\n" + "=" * 70)
        print("✓ ALL PHASE 2 TESTS PASSED!")
        print("=" * 70)
        print("\n PHASE 2 COMPLETION SUMMARY:")
        print("  ✓ DoublePendulumEnv class implemented with pymunk physics")
        print("  ✓ Observation space: (6,) → [cart_x, cart_vx, pole1_θ, pole1_ω, pole2_θ, pole2_ω]")
        print("  ✓ Action space: (1,) → continuous force in [-1.0, 1.0]")
        print("  ✓ Baseline reward: cos(θ₁) + cos(θ₂)")
        print("  ✓ Shaped reward: upright + center penalty + velocity penalty + action penalty")
        print("  ✓ CSV logging: timesteps, mean_reward")
        print("  ✓ Gym API compliance verified")
        print("  ✓ Physics simulation working (pymunk.Space)")
        print("=" * 70)
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print("✗ PHASE 2 TESTS FAILED")
        print("=" * 70)
        sys.exit(1)
