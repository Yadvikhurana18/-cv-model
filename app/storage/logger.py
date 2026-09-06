"""
Inspection Audit Logging and Defect Archiving Module.
Records all screening decisions, SSIM scores, defect types, and timestamps into CSV/JSON logs.
"""
import csv
from datetime import datetime
from pathlib import Path
import cv2
import pandas as pd
from app.config import config


class InspectionLogger:
    """Manages persistent CSV logging and image snapshot saving for quality compliance."""

    def __init__(self, log_file: Path | None = None):
        if log_file is None:
            log_file = config.logs_dir / "inspection_history.csv"

        self.log_file = Path(log_file)
        self.snapshots_dir = config.logs_dir / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self._init_csv()

    def _init_csv(self):
        if not self.log_file.exists():
            with open(self.log_file, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "component_id",
                    "verdict",
                    "cv_status",
                    "ml_status",
                    "ssim_score",
                    "defect_count",
                    "ml_confidence",
                    "summary",
                    "snapshot_path",
                ])

    def log_inspection(
        self,
        component_id: str,
        result: dict,
        annotated_frame: cv2.typing.MatLike | None = None,
    ) -> dict:
        """
        Appends inspection record to CSV and saves annotated snapshot if defective or requested.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        time_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        snapshot_rel_path = ""
        if annotated_frame is not None:
            snapshot_filename = f"{result['final_status']}_{component_id}_{time_tag}.jpg"
            snapshot_full_path = self.snapshots_dir / snapshot_filename
            cv2.imwrite(str(snapshot_full_path), annotated_frame)
            snapshot_rel_path = str(snapshot_full_path.relative_to(config.base_dir))

        row = [
            timestamp,
            component_id,
            result["final_status"],
            result["cv_status"],
            result["ml_status"],
            f"{result.get('ssim_score', 1.0):.3f}",
            result.get("defect_count", 0),
            f"{result.get('ml_confidence', 0.0):.3f}",
            result.get("summary", ""),
            snapshot_rel_path,
        ]

        with open(self.log_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        return {
            "timestamp": timestamp,
            "component_id": component_id,
            "verdict": result["final_status"],
            "snapshot_path": snapshot_rel_path,
        }

    def get_history_dataframe(self) -> pd.DataFrame:
        """Returns the inspection history as a pandas DataFrame."""
        if self.log_file.exists():
            try:
                return pd.read_csv(self.log_file)
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()
