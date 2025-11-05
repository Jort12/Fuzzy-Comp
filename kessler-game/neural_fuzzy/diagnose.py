#!/usr/bin/env python3
"""
Diagnostic script to test if your neural fuzzy model is working
"""
import os
import torch
import numpy as np

print("=" * 70)
print("NEURAL FUZZY MODEL DIAGNOSTIC")
print("=" * 70)

# Check if model files exist
model_dir = "models"
maneuver_path = os.path.join(model_dir, "maneuver.pt")
combat_path = os.path.join(model_dir, "combat.pt")

print("\n1. Checking model files...")
if os.path.exists(maneuver_path):
    size = os.path.getsize(maneuver_path)
    print(f"   ✅ maneuver.pt exists ({size:,} bytes)")
    if size < 1000:
        print(f"   ⚠️  WARNING: File is very small ({size} bytes)! Likely corrupted.")
else:
    print(f"   ❌ maneuver.pt NOT FOUND at {maneuver_path}")
    print(f"   ACTION: You need to train the model first!")
    print(f"   Run: python nf_train.py --task maneuver --num_mfs 3 --epochs 100")
    exit(1)

# Try to load the model
print("\n2. Loading model...")
try:
    bundle = torch.load(maneuver_path, map_location='cpu')
    print(f"   ✅ Model loaded successfully")
    print(f"   Task: {bundle.get('task', 'unknown')}")
    print(f"   Heads: {list(bundle.get('heads', {}).keys())}")
except Exception as e:
    print(f"   ❌ Failed to load model: {e}")
    exit(1)

# Check model structure
print("\n3. Checking model structure...")
for head_name, head_info in bundle['heads'].items():
    print(f"\n   Head: {head_name}")
    print(f"     - num_inputs: {head_info.get('num_inputs', 'unknown')}")
    print(f"     - num_mfs: {head_info.get('num_mfs', 'unknown')}")
    print(f"     - feature_cols: {head_info.get('feature_cols', 'unknown')}")
    
    # Check if state_dict has parameters
    state_dict = head_info.get('state_dict', {})
    if not state_dict:
        print(f"     ❌ ERROR: state_dict is empty!")
    else:
        print(f"     ✅ state_dict has {len(state_dict)} parameter tensors")
        
        # Check if parameters are all zeros
        all_zero = True
        for key, tensor in state_dict.items():
            if torch.abs(tensor).sum() > 0.001:
                all_zero = False
                break
        
        if all_zero:
            print(f"     ⚠️  WARNING: All parameters are near zero! Model didn't train properly.")
        else:
            print(f"     ✅ Parameters have non-zero values")

# Test prediction
print("\n4. Testing model prediction...")
try:
    from nf_infer import NFPolicy
    
    policy = NFPolicy(maneuver_path)
    
    # Create test input
    test_input = torch.zeros((1, 8)).to(policy.device)
    test_input[0, 0] = 200.0  # distance
    test_input[0, 1] = 10.0   # ttc
    test_input[0, 2] = 0.5    # heading_err
    test_input[0, 3] = 50.0   # approach_speed
    test_input[0, 4] = 0.0    # ammo
    test_input[0, 5] = 5.0    # mines
    test_input[0, 6] = 1.0    # threat_density
    test_input[0, 7] = 0.0    # threat_angle
    
    thrust, turn_rate = policy.act_maneuver_tensor(test_input)
    
    print(f"\n   Test prediction:")
    print(f"     Input: dist=200, ttc=10, heading_err=0.5, approach=50")
    print(f"     Output: thrust={thrust:.4f}, turn_rate={turn_rate:.4f}")
    
    if abs(thrust) < 0.01 and abs(turn_rate) < 0.01:
        print(f"\n   ❌ PROBLEM: Both outputs are near zero!")
        print(f"   This means the model didn't learn anything.")
        print(f"\n   LIKELY CAUSES:")
        print(f"     1. Model wasn't retrained after fixing nf_infer.py")
        print(f"     2. Training data wasn't in the right location")
        print(f"     3. Training failed but didn't show errors")
        print(f"\n   SOLUTION:")
        print(f"     1. Delete the model: rm models/maneuver.pt")
        print(f"     2. Verify training data exists:")
        print(f"        ls -lh data/maneuver.csv")
        print(f"     3. Retrain:")
        print(f"        python nf_train.py --task maneuver --num_mfs 3 --epochs 100 --lr 0.003")
        print(f"     4. Watch for decreasing loss values during training")
    else:
        print(f"\n   ✅ SUCCESS: Model is making predictions!")
        print(f"   Thrust: {thrust:.2f} (expected: -150 to 150)")
        print(f"   Turn: {turn_rate:.2f} (expected: -180 to 180)")
        
except ImportError:
    print(f"   ⚠️  Cannot import NFPolicy. Check if nf_infer.py exists.")
except Exception as e:
    print(f"   ❌ Error during prediction: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)