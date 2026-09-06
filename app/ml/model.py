"""
PyTorch Deep Learning Model Architecture for Electronic Component Defect Screening.
Implements ComponentDefectNet with Attention mechanism and Grad-CAM anomaly visualization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SpatialAttention(nn.Module):
    """Spatial attention module to focus on defect locations (e.g. pins, solder, cracks)."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.conv(scale)
        scale = self.sigmoid(scale)
        return x * scale


class ComponentDefectNet(nn.Module):
    """
    Component Defect Classification & Screening Neural Network.
    Supports Grad-CAM hooks to produce defect heatmaps on component images.
    """

    def __init__(
        self,
        num_classes: int = 2,
        backbone_name: str = "mobilenet_v3_large",
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone_name
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None

        try:
            if backbone_name == "mobilenet_v3_large":
                weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
                base = models.mobilenet_v3_large(weights=weights)
                self.features = base.features
                in_features = 960
            elif backbone_name == "resnet18":
                weights = models.ResNet18_Weights.DEFAULT if pretrained else None
                base = models.resnet18(weights=weights)
                self.features = nn.Sequential(
                    base.conv1,
                    base.bn1,
                    base.relu,
                    base.maxpool,
                    base.layer1,
                    base.layer2,
                    base.layer3,
                    base.layer4,
                )
                in_features = 512
            else:
                # Custom robust CNN backbone
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((7, 7)),
                )
                in_features = 256
        except Exception:
            # Safe offline fallback if weights download fails
            base = models.mobilenet_v3_large(weights=None)
            self.features = base.features
            in_features = 960

        # Spatial attention over feature maps
        self.attention = SpatialAttention(kernel_size=7)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Screening classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self._init_classifier_weights()

    def _init_classifier_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _hook_gradients(self, grad: torch.Tensor):
        self.gradients = grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        feats = self.features(x)
        
        # Save activations for Grad-CAM if requiring grad
        if x.requires_grad:
            self.activations = feats
            feats.register_hook(self._hook_gradients)

        feats_attended = self.attention(feats)
        pooled = self.pool(feats_attended)
        logits = self.classifier(pooled)
        return logits

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Runs inference and returns predicted classes and probability distributions.
        Returns: (pred_class, probabilities)
        """
        self.eval()
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds, probs

    def generate_gradcam(self, x: torch.Tensor, class_idx: int = 1) -> torch.Tensor:
        """
        Generates Grad-CAM defect heatmap for the specified class (default: 1 for Defective).
        Returns: (B, 1, H, W) normalized heatmap tensor.
        """
        self.eval()
        x = x.clone().detach().requires_grad_(True)
        logits = self.forward(x)
        score = logits[:, class_idx].sum()

        self.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Fallback if no gradients
            return torch.zeros((x.size(0), 1, x.size(2), x.size(3)), device=x.device)

        # Global average pool the gradients
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Resize CAM to match input image dimensions
        cam = F.interpolate(cam, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=False)

        # Min-max normalize per batch sample
        b, c, h, w = cam.shape
        cam_flat = cam.view(b, -1)
        cam_min = cam_flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        cam_max = cam_flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam
