import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models import BiRefNet
from .adapters import apply_conv2d, apply_linear


class LoRABiRefNet(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        rank: int = 8,
        alpha: float = 16.0
    ) -> None:
        super().__init__()
        self.model = model

        for p in self.model.parameters():
            p.requires_grad = False

        apply_linear(
            model = self.model.bb,
            rank = rank,
            alpha = alpha
        )
        apply_conv2d(
            model = self.model.decoder,
            rank = rank,
            alpha = alpha
        )

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.stats = {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable
        }

        self.aux_loss = torch.tensor(0.0)
        self.bce = nn.BCEWithLogitsLoss()

    def _train_step(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        scaled_preds, _ = self.model(x)
        (gdt_preds, gdt_labels), preds = scaled_preds

        loss = torch.tensor(0.0, device=x.device)
        for pred, label in zip(gdt_preds, gdt_labels):
            pred = F.interpolate(
                input = pred,
                size = label.shape[2:],
                mode = "bilinear",
                align_corners = True
            )
            label = label.sigmoid()
            loss = loss + self.bce(pred, label)

        self.aux_loss = loss
        return preds[-1]

    def _eval_step(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        preds = self.model(x)
        return preds[-1]

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        if self.training:
            return self._train_step(x)
        return self._eval_step(x)

    def get_adapter_params(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def save_adapters(
        self,
        path: str
    ) -> None:
        state = {
            k: v for k, v in self.state_dict().items()
            if "down" in k or "up" in k
        }
        torch.save(state, path)

    def load_adapters(
        self,
        path: str
    ) -> None:
        state = torch.load(
            path,
            map_location = "cpu",
            weights_only = True
        )
        self.load_state_dict(state, strict=False)
