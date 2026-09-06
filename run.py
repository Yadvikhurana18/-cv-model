"""
Unified Entrypoint for ISRO Component Screening & AI Inspection System.

Usage:
    python run.py --app               # Start all-in-one AI & Inspection GUI Hub
    python run.py                     # Default: Starts all-in-one AI & Inspection GUI Hub
    python run.py --mode api          # Run standalone FastAPI Server
    python run.py --mode train        # Run CLI GPU model training
    python run.py --mode test         # Run test suite
"""
import argparse
import subprocess
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def launch_gui():
    """Launches the Streamlit All-in-One GUI Hub."""
    print("=" * 65)
    print("🛰️ Starting A.R.G.U.S. - Autonomous Screening Hub...")
    print("   GUI URL: http://localhost:8501")
    print("=" * 65)
    dashboard_script = ROOT_DIR / "app" / "ui" / "dashboard.py"
    subprocess.run(["streamlit", "run", str(dashboard_script)], check=True)


def main():
    parser = argparse.ArgumentParser(description="A.R.G.U.S. - Autonomous Real-time Inspection & Screening Hub")
    parser.add_argument(
        "--app",
        action="store_true",
        help="Start the all-in-one interactive Web GUI (Live screening, GPU trainer, custom dataset manager)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="app",
        choices=["app", "gui", "train", "train-multiclass", "api", "generate", "test"],
        help="Execution mode (default: app)",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--port", type=int, default=8000, help="Port for API server")
    args = parser.parse_args()

    # If --app flag is passed or mode is app/gui
    if args.app or args.mode in ("app", "gui"):
        launch_gui()

    elif args.mode == "train":
        from app.ml.train import run_training
        print("[*] Launching GPU-accelerated PyTorch training (Binary Screening)...")
        run_training(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

    elif args.mode == "train-multiclass":
        from app.ml.train_multiclass import run_multiclass_training
        print("[*] Launching GPU-accelerated PyTorch training (6-Class Defect Diagnosis)...")
        run_multiclass_training(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

    elif args.mode == "api":
        import uvicorn
        print(f"[*] Starting FastAPI Inspection Server on http://127.0.0.1:{args.port}...")
        uvicorn.run("app.api.server:app", host="127.0.0.1", port=args.port, reload=False)

    elif args.mode == "generate":
        from app.dataset.generator import ComponentGenerator
        print("[*] Generating electronic component dataset...")
        gen = ComponentGenerator()
        gen.generate_dataset(num_normal=150, num_defective=150)
        gen.generate_multiclass_dataset(samples_per_class=100)

    elif args.mode == "test":
        print("[*] Running inspection pipeline test suite...")
        import unittest
        suite = unittest.defaultTestLoader.discover("tests")
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)


if __name__ == "__main__":
    main()
