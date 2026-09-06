"""
High-Performance GPU/CPU Inference Engine for Electronic Component Defect Screening.
Loads trained ComponentDefectNet weights, computes classification probabilities,
and generates Grad-CAM visual heatmaps.
"""
from pathlib import Path
import cv2
import numpy as np
import torch
from torchvision import transforms

from app.config import config
from app.ml.model import ComponentDefectNet


class DefectClassifier:
    """Inference wrapper for ComponentDefectNet."""

    def __init__(
        self,
        model_path: Path | None = None,
        device_name: str | None = None,
    ):
        if device_name:
            self.device = torch.device(device_name)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_path is None:
            model_path = config.best_model_path

        self.model_path = Path(model_path)
        self.model = ComponentDefectNet(
            num_classes=2,
            backbone_name=config.model_name,
            pretrained=False,
        ).to(self.device)

        self.is_loaded = False
        self._load_weights()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def _load_weights(self):
        if self.model_path.exists():
            try:
                try:
                    ckpt = torch.load(str(self.model_path), map_location=self.device, weights_only=True)
                except Exception:
                    ckpt = torch.load(str(self.model_path), map_location=self.device, weights_only=False)
                if "model_state_dict" in ckpt:
                    self.model.load_state_dict(ckpt["model_state_dict"])
                else:
                    self.model.load_state_dict(ckpt)
                self.model.eval()
                self.is_loaded = True
                print(f"[ML Engine] Loaded defect screening weights from {self.model_path}")
            except Exception as e:
                print(f"[ML Engine] Warning: Failed to load checkpoint: {e}")
                self.is_loaded = False
        else:
            print(f"[ML Engine] Model file {self.model_path} not found. Running with initialized weights.")
            self.is_loaded = False

    def predict(
        self, bgr_image: np.ndarray, compute_cam: bool = True
    ) -> dict:
        """
        Runs inference on an OpenCV BGR image.
        
        Returns:
            dict containing:
                - is_defective (bool)
                - label_name ("PASS" / "FAIL")
                - confidence (float)
                - defect_prob (float)
                - normal_prob (float)
                - cam_heatmap (np.ndarray or None)
                - overlay_image (np.ndarray)
        """
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            preds, probs = self.model.predict(tensor)
            pred_idx = preds[0].item()
            normal_prob = probs[0, 0].item()
            defect_prob = probs[0, 1].item()

        is_defective = bool(pred_idx == 1)
        label_name = "FAIL" if is_defective else "PASS"
        confidence = defect_prob if is_defective else normal_prob

        cam_heatmap = None
        overlay_image = bgr_image.copy()

        if compute_cam and is_defective:
            try:
                # Compute Grad-CAM for Defective class
                cam_tensor = self.model.generate_gradcam(tensor, class_idx=1)
                cam_np = cam_tensor[0, 0].cpu().detach().numpy()
                cam_np = cv2.resize(cam_np, (bgr_image.shape[1], bgr_image.shape[0]))
                cam_heatmap = (cam_np * 255).astype(np.uint8)
                
                # Apply ColorMap JET
                heatmap_color = cv2.applyColorMap(cam_heatmap, cv2.COLORMAP_JET)
                overlay_image = cv2.addWeighted(bgr_image, 0.65, heatmap_color, 0.35, 0)
            except Exception as e:
                print(f"[ML Engine] Grad-CAM error: {e}")

        return {
            "is_defective": is_defective,
            "status": label_name,
            "confidence": float(confidence),
            "normal_prob": float(normal_prob),
            "defect_prob": float(defect_prob),
            "cam_heatmap": cam_heatmap,
            "overlay_image": overlay_image,
        }
