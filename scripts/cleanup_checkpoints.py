#!/usr/bin/env python3
"""
Manual cleanup of checkpoints to enforce the new retention policy.

Policy:
1. Keep the last 5 checkpoints unconditionally.
2. For older checkpoints, keep only those where step % 100 == 0.
"""
from pathlib import Path
import re
import sys

def get_step(path):
    match = re.search(r'_step(\d+)\.pt$', str(path))
    return int(match.group(1)) if match else 0

def main():
    checkpoint_dir = Path("checkpoints")
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory {checkpoint_dir} does not exist.")
        sys.exit(1)
        
    print(f"Cleaning executing in: {checkpoint_dir.absolute()}")
    
    # Get all checkpoint files (exclude 'best' and 'final' checkpoints)
    all_ckpts = list(checkpoint_dir.glob("*_step*.pt"))
    all_ckpts.sort(key=get_step)
    
    if not all_ckpts:
        print("No checkpoints found.")
        return

    keep_last = 5
    thin_interval = 1000
    
    # Split
    if len(all_ckpts) <= keep_last:
        print(f"Only {len(all_ckpts)} checkpoints found. Keeping all (<= {keep_last}).")
        return
        
    recent_ckpts = all_ckpts[-keep_last:]
    older_ckpts = all_ckpts[:-keep_last]
    
    print(f"Total checkpoints: {len(all_ckpts)}")
    print(f"Recent (keeping): {[get_step(p) for p in recent_ckpts]}")
    
    deleted_count = 0
    preserved_count = 0
    
    for ckpt in older_ckpts:
        step = get_step(ckpt)
        
        # Logic: If step is NOT a multiple of 100, delete it
        if step % thin_interval != 0:
            print(f"Deleting: {ckpt.name} (step {step})")
            ckpt.unlink()
            deleted_count += 1
        else:
            print(f"Keeping:  {ckpt.name} (step {step}) - Milestone")
            preserved_count += 1
            
    print("-" * 40)
    print(f"Cleanup complete.")
    print(f"Deleted: {deleted_count}")
    print(f"Preserved (older): {preserved_count}")
    print(f"Preserved (recent): {len(recent_ckpts)}")
    print(f"Total remaining: {preserved_count + len(recent_ckpts)}")

if __name__ == "__main__":
    main()
