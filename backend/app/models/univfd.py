from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn as nn
from app.models.base import BaseDetector, ModelOutput


class UnivFDDetector(BaseDetector):
    """
    UnivFD: CLIP ViT-L/14 frozen + linear probe.
    Weights file contains only the linear layer; CLIP loaded via open_clip.
    Visualization: attention rollout across all transformer blocks.
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
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        self.clip_model.eval().to(device)
        for p in self.clip_model.parameters():
            p.requires_grad = False

        self.linear = nn.Linear(768, 1)
        state = torch.load(weights_path, map_location=device)
        # Handle both raw state_dict and wrapped checkpoints
        if isinstance(state, dict) and "fc.weight" in state:
            self.linear.weight.data = state["fc.weight"]
            self.linear.bias.data = state["fc.bias"]
        elif isinstance(state, dict) and "weight" in state:
            self.linear.load_state_dict(state)
        else:
            self.linear.load_state_dict(state)
        self.linear.eval().to(device)
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
        elapsed = (time.time() - t0) * 1000
        return ModelOutput(
            model_name=self.model_name,
            fake_prob=fake_prob,
            real_prob=1.0 - fake_prob,
            inference_time_ms=elapsed,
            features=features.cpu().numpy().flatten(),
        )

    def get_heatmap(self, preprocessed: dict) -> np.ndarray | None:
        """Attention rollout: Abnar & Zuidema 2020."""
        attentions: list[torch.Tensor] = []
        hooks = []

        def make_hook():
            def hook(module, inp, out):
                # out: (batch, heads, seq, seq) or tuple
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
                # Average over heads
                a = attn.mean(dim=1).squeeze(0)
                a = 0.5 * a + 0.5 * torch.eye(a.size(0))
                a = a / (a.sum(dim=-1, keepdim=True) + 1e-8)
                result = a @ result

            # CLS token → patches: result[0, 1:] → (196,) → (14, 14)
            mask = result[0, 1:].reshape(14, 14).numpy()
            mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
            return mask.astype(np.float32)
        except Exception:
            for h in hooks:
                h.remove()
            return None
