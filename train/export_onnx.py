import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO

class OnnxActor(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.mlp_extractor = policy.mlp_extractor.policy_net
        self.action_net = policy.action_net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.mlp_extractor(obs)
        action = self.action_net(features)
        return action

def export_policy_to_onnx(checkpoint_path: str, output_path: str, verify: bool = True):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print(f"Loading trained model from {checkpoint_path}...")
    model = PPO.load(checkpoint_path, device="cpu")
    actor = OnnxActor(model.policy)
    actor.eval()

    dummy_input = torch.randn(1, 18, dtype=torch.float32)

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
        dynamic_axes={"obs": {0: "batch_size"}, "action": {0: "batch_size"}}
    )
    print("ONNX export completed.")

    if verify:
        import onnx
        import onnxruntime as ort

        print("Verifying exported ONNX model...")
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)

        ort_session = ort.InferenceSession(output_path)
        test_inputs = np.random.randn(5, 18).astype(np.float32)

        with torch.no_grad