import os

# Security: Allow loading of local models (pickle)
os.environ["TRUST_REMOTE_CODE"] = "1"

from typing import Any

import matplotlib.cm as cm
import numpy as np
import torch

# --- Anomalib / Torch ---
from anomalib.deploy.inferencers.torch_inferencer import TorchInferencer
from loguru import logger
from PIL import Image

# --- SAM2 ---
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False


class IntegratedEngine:
    """Integrated AI Engine for Surface Anomaly Detection.

    This engine combines Anomalib PatchCore for surface anomaly detection and
    SAM2 (Segment Anything Model 2) for precise boundary segmentation of defects.
    """

    def __init__(self, anomalib_path: str, sam2_checkpoint: str, sam2_config: str | None = None) -> None:
        """Initializes the Integrated AI Engine with models.

        Args:
            anomalib_path: File path to the trained Anomalib model weight.
            sam2_checkpoint: File path to the SAM2 model checkpoint.
            sam2_config: Optional configuration file path for SAM2.
        """
        self.device = self._get_device()
        self.load_error: str | None = None

        logger.info(f"Initializing IntegratedEngine on device: {self.device}")

        # 1. Anomalib Load
        try:
            logger.info(f"Loading Anomalib model from: {anomalib_path}")
            self.anomalib_engine = TorchInferencer(path=anomalib_path, device=self.device)
            logger.info("Anomalib model loaded successfully.")
        except Exception as e:
            self.load_error = f"Anomalib Load Error: {e}"
            logger.error(self.load_error)
            raise RuntimeError(self.load_error) from e

        # 2. SAM2 Load - Auto-detect config if not provided
        if sam2_config is None:
            sam2_config = self._guess_sam2_config(sam2_checkpoint)

        self.sam2_predictor = None
        if not SAM2_AVAILABLE:
            self.load_error = "SAM2 library not installed (ImportError)"
            logger.warning(self.load_error)
        else:
            try:
                logger.info(f"Loading SAM2 with config: {sam2_config}, checkpoint: {sam2_checkpoint}")
                sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=self.device)
                self.sam2_predictor = SAM2ImagePredictor(sam2_model)
                logger.info("SAM2 model loaded successfully.")
            except Exception as e:
                import traceback

                self.load_error = f"SAM2 Load Error: {e}\n{traceback.format_exc()}"
                logger.error(f"SAM2 failed to load: {self.load_error}")

    def _guess_sam2_config(self, checkpoint_path: str) -> str:
        """Guesses the configuration file name based on SAM2 checkpoint path.

        Args:
            checkpoint_path: Path to the SAM2 checkpoint.

        Returns:
            The configuration filename string.
        """
        name = os.path.basename(checkpoint_path).lower()
        if "tiny" in name:
            return "sam2_hiera_t.yaml"
        if "small" in name:
            return "sam2_hiera_s.yaml"
        if "base_plus" in name:
            return "sam2_hiera_b+.yaml"
        if "large" in name:
            return "sam2_hiera_l.yaml"
        return "sam2_hiera_t.yaml"  # Default

    def _get_device(self) -> str:
        """Determines the most optimal torch device available.

        Returns:
            Optimal device identifier string (cuda, mps, or cpu).
        """
        if torch.cuda.is_available():
            try:
                torch.zeros(1).cuda()
                logger.debug("CUDA acceleration is active.")
                return "cuda"
            except Exception as e:
                logger.debug(f"CUDA is present but test allocation failed: {e}")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.debug("Metal Performance Shaders (MPS) acceleration is active.")
            return "mps"
        logger.debug("CPU is active for PyTorch computation.")
        return "cpu"

    def analyze_anomalib(self, image: Image.Image) -> dict[str, Any]:
        """Runs PatchCore inference and returns heatmap and peak anomaly coordinate.

        Args:
            image: A PIL Image containing the target surface.

        Returns:
            A dictionary containing:
                - heatmap: Numpy array of the anomaly heatmap resized to original dimensions.
                - score: Float representing the computed anomaly score.
                - peak_point: Tuple (x, y) coordinates of the highest anomaly intensity.
        """
        logger.info("Running Anomalib PatchCore inference...")
        img_arr = np.array(image.convert("RGB"))
        h, w = img_arr.shape[:2]

        from torchvision import transforms

        preprocess = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        img_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if hasattr(self.anomalib_engine.model, "model"):
                prediction = self.anomalib_engine.model.model(img_tensor)
            else:
                prediction = self.anomalib_engine.model(img_tensor)

        # Extract results
        if isinstance(prediction, dict):
            heat_map = prediction.get("anomaly_map", None)
            pred_score = prediction.get("pred_score", None)
        else:
            heat_map = getattr(prediction, "anomaly_map", None)
            pred_score = getattr(prediction, "pred_score", None)

        if isinstance(heat_map, torch.Tensor):
            heat_map = heat_map.squeeze().cpu().numpy()
        else:
            heat_map = np.zeros((256, 256), dtype=np.float32)
            logger.warning("Anomaly map was not returned properly by Anomalib engine.")

        if isinstance(pred_score, torch.Tensor):
            pred_score = pred_score.item()
        elif pred_score is None:
            pred_score = 0.0
            logger.warning("Pred score was not returned properly by Anomalib engine.")

        # Find Peak Point (Max intensity in heatmap)
        # Resize heatmap to original image size for accurate coordinate
        heatmap_resized = np.array(Image.fromarray(heat_map).resize((w, h), resample=Image.BICUBIC))

        # Safe normalization for coordinate argmax
        peak_y, peak_x = np.unravel_index(np.argmax(heatmap_resized), heatmap_resized.shape)

        logger.info(f"Anomalib inference complete. Score: {pred_score:.4f}, Peak: ({peak_x}, {peak_y})")
        return {
            "heatmap": heatmap_resized,
            "score": float(pred_score),
            "peak_point": (int(peak_x), int(peak_y)),
        }

    def segment_with_sam2(self, image: Image.Image, points: np.ndarray, labels: np.ndarray) -> np.ndarray | None:
        """Runs SAM2 segmentation based on point prompts to isolate anomaly area.

        Args:
            image: A PIL Image of the target surface.
            points: A Numpy array of prompt coordinate points shape (N, 2).
            labels: A Numpy array of prompt labels shape (N,).

        Returns:
            A binary mask Numpy array of defect shape (H, W), or None if SAM2 is unavailable.
        """
        if self.sam2_predictor is None:
            logger.warning("SAM2 Predictor is unavailable. Skipping segmentation.")
            return None

        logger.info(f"Running SAM2 segmentation on point prompt: {points[0]}")
        img_arr = np.array(image.convert("RGB"))

        try:
            self.sam2_predictor.set_image(img_arr)
            masks, scores, logits = self.sam2_predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=False,
            )

            if masks is not None and len(masks) > 0:
                mask_sum: float = np.sum(masks[0])
                logger.info(f"SAM2 segmentation complete. Target mask area sum: {mask_sum} pixels.")
                return masks[0]
            else:
                logger.warning("SAM2 did not return any binary mask.")
                return None
        except Exception as e:
            logger.error(f"Error during SAM2 prediction: {e}")
            return None

    def create_overlay(
        self,
        image: Image.Image,
        mask: np.ndarray,
        color: tuple[int, int, int] = (255, 0, 0),
        alpha: float = 0.5,
    ) -> Image.Image:
        """Overlays a binary mask onto a PIL Image.

        Args:
            image: Base PIL Image.
            mask: Binary mask Numpy array (values 0 or 1).
            color: RGB color tuple to draw the mask.
            alpha: Transparency factor (0.0 to 1.0).

        Returns:
            An overlaid PIL Image.
        """
        logger.debug("Creating mask overlay on PIL image.")
        img_arr = np.array(image.convert("RGB"))
        mask_overlay = np.zeros_like(img_arr)
        mask_overlay[mask > 0] = color

        combined = (img_arr * (1 - alpha) + mask_overlay * alpha).astype(np.uint8)
        return Image.fromarray(combined)

    def create_heatmap_overlay(self, image: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
        """Overlays a continuous anomaly heatmap onto a PIL Image.

        Args:
            image: Base PIL Image.
            heatmap: Continuous continuous intensity heatmap array.
            alpha: Transparency factor (0.0 to 1.0).

        Returns:
            A PIL Image overlaid with Jet colormap.
        """
        logger.debug("Creating heatmap overlay on PIL image.")
        img_arr = np.array(image.convert("RGB"))

        # Normalize heatmap safely
        h_min, h_max = heatmap.min(), heatmap.max()
        denom = h_max - h_min
        if denom < 1e-8:
            denom = 1e-8
        heatmap_norm = (heatmap - h_min) / denom

        colormap = cm.get_cmap("jet")
        heatmap_colored = (colormap(heatmap_norm)[:, :, :3] * 255).astype(np.uint8)

        overlay = (img_arr * (1 - alpha) + heatmap_colored * alpha).astype(np.uint8)
        return Image.fromarray(overlay)
