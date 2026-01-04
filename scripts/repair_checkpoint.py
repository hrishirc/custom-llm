#!/usr/bin/env python3
"""Repair scheduler state in checkpoints to match global_step.

The scheduler was reset multiple times due to a bug where load_checkpoint
didn't restore scheduler state (scheduler was None at load time).

This script patches the latest checkpoint so that last_epoch = global_step,
allowing training to resume with the correct learning rate.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import shutil
import re


def repair_checkpoint(checkpoint_path: Path) -> None:
    """Repair scheduler state to match global_step."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    global_step = ckpt["state"]["global_step"]
    sched_state = ckpt.get("scheduler_state_dict", {})
    last_epoch = sched_state.get("last_epoch", 0)
    
    print(f"Current state:")
    print(f"  global_step: {global_step}")
    print(f"  last_epoch:  {last_epoch}")
    print(f"  Mismatch:    {global_step - last_epoch} steps")
    
    if global_step == last_epoch:
        print("No repair needed - scheduler is in sync!")
        return
    
    # Backup original
    backup_path = checkpoint_path.with_suffix(".pt.backup")
    if not backup_path.exists():
        shutil.copy(checkpoint_path, backup_path)
        print(f"Backed up to: {backup_path}")
    
    # Repair scheduler state
    sched_state["last_epoch"] = global_step
    sched_state["_step_count"] = global_step + 1
    # Note: _last_lr will be recalculated by the scheduler on next step
    
    ckpt["scheduler_state_dict"] = sched_state
    
    # Save repaired checkpoint
    torch.save(ckpt, checkpoint_path)
    print(f"Repaired! New last_epoch = {global_step}")


def main():
    checkpoint_dir = Path("checkpoints")
    
    # Find latest checkpoint
    ckpts = list(checkpoint_dir.glob("phase1_grammar_step*.pt"))
    if not ckpts:
        print("No checkpoints found!")
        return
    
    def get_step(p):
        m = re.search(r'_step(\d+)\.pt$', str(p))
        return int(m.group(1)) if m else 0
    
    latest = max(ckpts, key=get_step)
    repair_checkpoint(latest)


if __name__ == "__main__":
    main()
