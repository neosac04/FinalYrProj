from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn as nn
from app.models.base import BaseDetector, ModelOutput


class UnivFDDetector(BaseDetector):
    """
    UnivFD: CLIP ViT-L/14 frozen + linear probe.
    Loads ONLY linear layer weights (weight, bias).
    Includes stability + debug logging.
    """

    model_name = "univfd"

    def __init__(self) -> None:
        self._loaded = False
        self.clip_model = None
        self.linear: nn.Linear | None = None
        self.device = torch.device("cpu")

    def load(self, weights_path: str, device: torch.device) -> None:
        import open_clip

        self.device = device

        print(f"\n🔍 Loading UnivFD from: {weights_path}")

        # 🔹 Load CLIP backbone
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        self.clip_model.eval().to(device)

        for p in self.clip_model.parameters():
            p.requires_grad = False

        # 🔹 Create linear probe
        self.linear = nn.Linear(768, 1).to(device)

        # 🔹 Load weights (direct mapping)
        state = torch.load(weights_path, map_location=device)

        if not isinstance(state, dict) or "weight" not in state:
            raise ValueError("❌ Invalid UnivFD checkpoint format")

        # Load directly
        self.linear.weight.data.copy_(state["weight"])
        self.linear.bias.data.copy_(state["bias"])

        print("✅ UnivFD loaded")
        print(f"Weight shape: {state['weight'].shape}")
        print(f"Bias shape: {state['bias'].shape}")

        self.linear.eval()
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, preprocessed: dict) -> ModelOutput:
        t0 = time.time()

        with torch.no_grad():
            tensor = preprocessed["clip_tensor"].unsqueeze(0).to(self.device)

            features = self.clip_model.encode_image(tensor)
            features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)

            logit = self.linear(features.float())
            fake_prob = torch.sigmoid(logit).item()
            print("Fake probability:", fake_prob)

        # 🔥 Stability clamp
        fake_prob = max(min(fake_prob, 0.999), 0.001)
        real_prob = 1.0 - fake_prob

        elapsed = (time.time() - t0) * 1000

        return ModelOutput(
            model_name=self.model_name,
            fake_prob=fake_prob,
            real_prob=real_prob,
            inference_time_ms=elapsed,
            features=features.cpu().numpy().flatten(),
        )

    def get_heatmap(self, preprocessed: dict) -> np.ndarray | None:
        """Attention rollout: Abnar & Zuidema 2020."""
        attentions: list[torch.Tensor] = []
        hooks = []

        def make_hook():
            def hook(module, inp, out):
                if isinstance(out, tuple):
                    out = out[0]
                attentions.append(out.detach().cpu())
            return hook

        try:
            for block in self.clip_model.visual.transformer.resblocks:
                hooks.append(block.attn.register_forward_hook(make_hook()))

            with torch.no_grad():
                tensor = preprocessed["clip_tensor"].unsqueeze(0).to(self.device)
                self.clip_model.encode_image(tensor)

            for h in hooks:
                h.remove()

            if not attentions:
                return None

            result = torch.eye(attentions[0].size(-1))

            for attn in attentions:
                a = attn.mean(dim=1).squeeze(0)
                a = 0.5 * a + 0.5 * torch.eye(a.size(0))
                a = a / (a.sum(dim=-1, keepdim=True) + 1e-8)
                result = a @ result

            mask = result[0, 1:].reshape(14, 14).numpy()
            mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

            return mask.astype(np.float32)

        except Exception:
            for h in hooks:
                h.remove()
            return None