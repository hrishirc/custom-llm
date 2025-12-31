#!/usr/bin/env python3
"""Initialize SQLite training state database.

Creates the database with:
- Schema for downloads, stages, and training phases
- Initial state with Wikipedia marked as downloaded

Usage:
    python scripts/init_db.py
    python scripts/init_db.py --db-path data/my_state.db
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.workflows.state_db import StateDB, init_default_state


def main():
    parser = argparse.ArgumentParser(description="Initialize training state database")
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/training_state.db",
        help="Path to SQLite database"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing database"
    )
    
    args = parser.parse_args()
    
    db_path = Path(args.db_path)
    
    if db_path.exists() and not args.force:
        print(f"Database already exists: {db_path}")
        print("Use --force to overwrite")
        
        # Show current state
        db = StateDB(str(db_path))
        state = db.get_full_state()
        
        print("\nCurrent state:")
        print(f"  Downloads: {len(state.get('downloads', {}))} tracked")
        print(f"  Phases: {len(state.get('phases', {}))} tracked")
        
        for dataset, info in state.get("downloads", {}).items():
            print(f"    - {dataset}: {info.get('status', 'unknown')}")
        
        return
    
    if db_path.exists() and args.force:
        db_path.unlink()
        print(f"Deleted existing database: {db_path}")
    
    # Create and initialize
    print(f"Creating database: {db_path}")
    db = StateDB(str(db_path))
    init_default_state(db)
    
    print("\nInitialized with:")
    print("  Downloads:")
    print("    - wikipedia: complete (18 GB)")
    print("    - pg19: pending")
    print("    - bookcorpus: pending")
    print("    - pubmed: pending")
    print("    - philpapers: pending")
    print("")
    print("  Training phases:")
    print("    - Phase 1 (Grammar): 50,000 steps pending")
    print("    - Phase 2 (Vocabulary): 12,500 steps pending")
    print("    - Phase 2b (Scientific): 6,250 steps pending")
    print("")
    print(f"Database ready: {db_path}")


if __name__ == "__main__":
    main()
