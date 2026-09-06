"""
Test GUI training callback and pipeline reload.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ml.train import run_training
from app.ml.train_multiclass import run_multiclass_training


def test_callback():
    print("Testing binary training callback...")
    events = []

    def on_prog(curr, total, rec):
        events.append(rec)
        print(f"  [Epoch {curr}/{total}] Loss: {rec['train_loss']:.4f} | Val Acc: {rec['val_acc']*100:.1f}%")

    res = run_training(epochs=2, batch_size=16, progress_callback=on_prog, auto_generate_if_empty=False)
    assert len(events) == 2, f"Expected 2 events, got {len(events)}"
    print("[*] Binary Training Callback Test Passed! Best Acc: " + str(round(res['best_val_acc']*100, 1)) + "%\n")

    print("Testing multi-class training callback...")
    events_mc = []

    def on_prog_mc(curr, total, rec):
        events_mc.append(rec)
        print(f"  [MC Epoch {curr}/{total}] Loss: {rec['train_loss']:.4f} | Val Acc: {rec['val_acc']*100:.1f}%")

    res_mc = run_multiclass_training(epochs=2, batch_size=16, progress_callback=on_prog_mc, auto_generate_if_empty=False)
    assert len(events_mc) == 2, f"Expected 2 events, got {len(events_mc)}"
    print(f"[*] Multi-Class Training Callback Test Passed! Best Acc: {res_mc['best_acc']*100:.1f}%\n")


if __name__ == "__main__":
    test_callback()
