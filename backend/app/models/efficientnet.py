from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from app.models.base import BaseDetector, ModelOutput


class EfficientNetDetector(BaseDetector):
    """
    EfficientNet-B4 trained on FaceForensics++ (DeepfakeBench v1.0.1).
    Uses the efficientnet_pytorch (lukemelas) package to match DeepfakeBench's
    checkpoint format.  Key remapping: backbone.* → * and last_layer → _fc.
    GradCAM++ hooks on the last conv block (_conv_head).
    """

    model_name = "efficientnet"

    def __init__(self) -> None:
        self._loaded = False
        self.model: nn.Module | None = None
        self.device = torch.device("cpu")
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

    def load(self, weights_path: str, device: torch.device) -> None:
        from efficientnet_pytorch import EfficientNet

        self.device = device

        # Build backbone matching DeepfakeBench's EfficientNetB4Detector
        backbone = EfficientNet.from_name("efficientnet-b4")
        in_feats = backbone._fc.in_features
        backbone._fc = nn.Linear(in_feats, 2)
        self.model = backbone

        state = torch.load(weights_path, map_location=device)
        cleaned: dict = {}
        for k, v in state.items():
            k2 = k.replace("backbone.efficientnet.", "")
            k2 = k2.replace("backbone.", "")   # strip bare backbone. for last_layer.*
            k2 = k2.replace("last_layer.", "_fc.")
            cleaned[k2] = v

        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        self.model.eval().to(device)

        # GradCAM++ target: final conv before global pool
        self._gradcam_layer = self.model._conv_head
        self._register_hooks()
        self._loaded = True

    def _register_hooks(self) -> None:
        def fwd_hook(module, inp, out):
            self._activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self._gradcam_layer.register_forward_hook(fwd_hook)
        self._gradcam_layer.register_full_backward_hook(bwd_hook)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, preprocessed: dict) -> ModelOutput:
        t0 = time.time()
        with torch.no_grad():
            tensor = preprocessed["imagenet_tensor"].unsqueeze(0).to(self.device)
            tensor = F.interpolate(tensor, size=(380, 380), mode="bilinear", align_corners=False)
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1).squeeze()
        elapsed = (time.time() - t0) * 1000
        return ModelOutput(
            model_name=self.model_name,
            fake_prob=float(probs[1]),
            real_prob=float(probs[0]),
            inference_time_ms=elapsed,
            features=logits.cpu().numpy().flatten(),
        )

    def get_heatmap(self, preprocessed: dict) -> np.ndarray | None:
        """GradCAM++ for fake class (index 1)."""
        try:
            tensor = preprocessed["imagenet_tensor"].unsqueeze(0).to(self.device)
            tensor = F.interpolate(tensor, size=(380, 380), mode="bilinear", align_corners=False)
            tensor = tensor.detach().requires_grad_(True)

            logits = self.model(tensor)
            self.model.zero_grad()
            one_hot = torch.zeros_like(logits)
            one_hot[0, 1] = 1.0
            logits.backward(gradient=one_hot, retain_graph=False)

            acts = self._activations
            grads = self._gradients
            if acts is None or grads is None:
                return None

            grad_sq = grads ** 2
            grad_cu = grads ** 3
            sum_acts = acts.sum(dim=(2, 3), keepdim=True)
            denom = 2 * grad_sq + sum_acts * grad_cu
            denom = torch.where(denom != 0, denom, torch.ones_like(denom))
            alpha = grad_sq / denom
            weights = (alpha * torch.relu(grads)).sum(dim=(2, 3), keepdim=True)
            cam = (weights * acts).sum(dim=1).squeeze()
            cam = torch.relu(cam).detach().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            return cam.astype(np.float32)
        except Exception:
            return None
