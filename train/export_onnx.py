#!/usr/bin/env python3
"""Export a trained SB3 PPO policy to ONNX for the ROS 2 inference node.

The exported graph is obs -> deterministic action, with the action-space clip
baked in so ONNX Runtime reproduces `model.predict(deterministic=True)`.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO

OBS_DIM = 18
ACT_DIM = 2


class OnnxActor(nn.Module):
    """Deterministic actor: features -> policy MLP -> action mean -> clip."""

    def __init__(self, policy, low: np.ndarray, high: np.ndarray):
        super().__init__()
        self.mlp_extractor = policy.mlp_extractor.policy_net
        self.action_net = policy.action_net
        self.register_buffer("low", torch.as_tensor(low, dtype=torch.float32))
        self.register_buffer("high", torch.as_tensor(high, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        action = self.action_net(self.mlp_extractor(obs))
        return torch.minimum(torch.maximum(action, self.low), self.high)


def build_actor(model: PPO) -> OnnxActor:
    if model.policy.squash_output:
        raise ValueError("squashed policies need tanh in the graph; not supported")
    low = model.action_space.low.astype(np.float32)
    high = model.action_space.high.astype(np.float32)
    actor = OnnxActor(model.policy, low, high)
    actor.eval()
    return actor


def export_policy_to_onnx(checkpoint_path: str, output_path: str, verify: bool = True):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print(f"Loading trained model from {checkpoint_path}...")
    model = PPO.load(checkpoint_path, device="cpu")
    if model.observation_space.shape != (OBS_DIM,):
        raise ValueError(f"expected obs {(OBS_DIM,)}, got {model.observation_space.shape}")
    if model.action_space.shape != (ACT_DIM,):
        raise ValueError(f"expected act {(ACT_DIM,)}, got {model.action_space.shape}")

    actor = build_actor(model)
    dummy_input = torch.zeros(1, OBS_DIM, dtype=torch.float32)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"Exporting to ONNX format at {output_path}...")
    torch.onnx.export(
        actor,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch_size"}, "action": {0: "batch_size"}},
        dynamo=False,
    )
    print("ONNX export completed.")

    if verify:
        verify_export(model, output_path)


def verify_export(model: PPO, output_path: str, n_samples: int = 256):
    """ONNX Runtime output must match SB3 `predict(deterministic=True)`."""
    import onnx
    import onnxruntime as ort

    print("Verifying exported ONNX model...")
    onnx.checker.check_model(onnx.load(output_path))

    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    obs = rng.uniform(-1.0, 1.0, size=(n_samples, OBS_DIM)).astype(np.float32)

    sb3_out, _ = model.predict(obs, deterministic=True)
    onnx_out = session.run(None, {session.get_inputs()[0].name: obs})[0]

    max_diff = float(np.max(np.abs(sb3_out - onnx_out)))
    print(f"Max |SB3 predict - ONNX Runtime| over {n_samples} samples: {max_diff:.3e}")
    if max_diff >= 1e-5:
        raise AssertionError(f"ONNX/PyTorch mismatch: {max_diff:.3e} >= 1e-5")

    low, high = model.action_space.low, model.action_space.high
    if not (np.all(onnx_out >= low - 1e-6) and np.all(onnx_out <= high + 1e-6)):
        raise AssertionError("ONNX output escapes the action bounds")
    print("Verification successful: ONNX matches SB3 and respects action bounds.")


def main():
    parser = argparse.ArgumentParser(description="Export a trained SB3 PPO model to ONNX.")
    parser.add_argument("--ckpt", default="runs/phase2_dr/latest.zip")
    parser.add_argument("--out", default="../src/drone_robot/models/drone_brain.onnx")
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    export_policy_to_onnx(args.ckpt, args.out, verify=not args.no_verify)


if __name__ == "__main__":
    main()
