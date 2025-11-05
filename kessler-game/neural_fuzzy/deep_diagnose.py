#!/usr/bin/env python3
"""
Deep diagnostic - Check raw model outputs and normalization step-by-step
"""
import sys
import torch
import numpy as np

print("=" * 70)
print("DEEP DIAGNOSTIC - Raw Model Output Analysis")
print("=" * 70)

# Force fresh import
if 'nf_infer' in sys.modules:
    del sys.modules['nf_infer']

from nf_infer import NFPolicy

# Load model
print("\n1. Loading model...")
policy = NFPolicy("models/maneuver.pt")
print(f"   Device: {policy.device}")
print(f"   Models loaded: {list(policy.models.keys())}")

# Get normalization parameters
thrust_model, thrust_mu, thrust_sd = policy.models['thrust']
print(f"\n2. Thrust model normalization:")
print(f"   mu (means): {thrust_mu[:3]}..." if thrust_mu else "   mu: None")
print(f"   sd (stdevs): {thrust_sd[:3]}..." if thrust_sd else "   sd: None")

# Create test input - RAW features (not normalized)
print("\n3. Creating test input...")
test_features = {
    'dist': 200.0,
    'ttc': 10.0,
    'heading_err': 0.5,
    'approach_speed': 50.0,
    'ammo': 0.0,
    'mines': 5.0,
    'threat_density': 1.0,
    'threat_angle': 0.0
}
print(f"   Raw features: dist={test_features['dist']}, ttc={test_features['ttc']}")

# Build input tensor (RAW, will be normalized by prep())
x_raw = torch.zeros((1, 8)).to(policy.device)
feature_names = ['dist', 'ttc', 'heading_err', 'approach_speed', 'ammo', 'mines', 'threat_density', 'threat_angle']
for i, fname in enumerate(feature_names):
    x_raw[0, i] = test_features[fname]

print(f"   Raw input tensor: {x_raw[0, :3].cpu().numpy()}")

# Normalize manually to see what model receives
if thrust_mu is not None:
    mu_tensor = torch.tensor(thrust_mu, dtype=torch.float32).to(policy.device)
    sd_tensor = torch.tensor(thrust_sd, dtype=torch.float32).to(policy.device)
    sd_tensor[sd_tensor < 1e-6] = 1.0
    x_normalized = (x_raw - mu_tensor) / sd_tensor
    print(f"   Normalized input: {x_normalized[0, :3].cpu().numpy()}")

# Get RAW model output (before any activation)
print("\n4. Testing THRUST model...")
with torch.no_grad():
    y_raw = thrust_model(x_raw).squeeze().item()
    print(f"   Raw model output (y_t): {y_raw:.6f}")
    
    # Apply tanh
    thrust_norm = np.tanh(y_raw)
    print(f"   After tanh(y_t): {thrust_norm:.6f}")
    
    # Denormalize
    thrust_final = thrust_norm * 150.0
    print(f"   After × 150: {thrust_final:.6f}")
    
    # Clamp
    thrust_clamped = max(-150.0, min(150.0, thrust_final))
    print(f"   After clamp: {thrust_clamped:.6f}")

# Test turn_rate too
print("\n5. Testing TURN_RATE model...")
turn_model, turn_mu, turn_sd = policy.models['turn_rate']
with torch.no_grad():
    y_raw = turn_model(x_raw).squeeze().item()
    print(f"   Raw model output (y_r): {y_raw:.6f}")
    
    turn_norm = np.tanh(y_raw)
    print(f"   After tanh(y_r): {turn_norm:.6f}")
    
    turn_final = turn_norm * 180.0
    print(f"   After × 180: {turn_final:.6f}")

# Now test using the actual method
print("\n6. Testing act_maneuver_tensor() method...")
thrust, turn = policy.act_maneuver_tensor(x_raw)
print(f"   Thrust: {thrust:.6f}")
print(f"   Turn: {turn:.6f}")

# Analysis
print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)

if abs(y_raw) < 0.001:
    print("❌ PROBLEM: Raw model output is near zero!")
    print("   This means the model parameters are all zeros or very small.")
    print("   The model didn't train properly.")
    print("\n   SOLUTION: Retrain the model with better parameters:")
    print("   1. Delete old model: rm models/maneuver.pt")
    print("   2. Retrain with more MFs: python nf_train.py --task maneuver --num_mfs 4 --epochs 150 --lr 0.005")
elif abs(thrust) < 0.001 and abs(y_raw) > 0.001:
    print("❌ PROBLEM: Model outputs non-zero but final thrust is zero!")
    print("   This is a bug in act_maneuver_tensor() method.")
    print("   Check if the correct nf_infer.py is being used.")
    print(f"\n   Check: grep 'np.tanh' nf_infer.py")
elif abs(thrust) > 1.0:
    print("✅ SUCCESS: Model is working!")
    print(f"   Thrust: {thrust:.2f}")
    print(f"   Turn: {turn:.2f}")
    print("\n   Your ship will move in the game!")
else:
    print("⚠️  Model output is very small but non-zero")
    print("   The model might be undertrained or too conservative.")
    print(f"   Try retraining with: --num_mfs 4 --epochs 200 --lr 0.01")

print("=" * 70)