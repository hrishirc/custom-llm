#!/usr/bin/env python3
"""CLI for starting and managing LLM training workflows.

Usage:
    # Start training (all phases)
    python scripts/start_training.py start
    
    # Start specific phases
    python scripts/start_training.py start --phases 1,2
    
    # Check workflow status
    python scripts/start_training.py status my-training-run
    
    # Cancel a workflow
    python scripts/start_training.py cancel my-training-run
"""
import asyncio
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from temporalio.client import Client
from temporalio.common import WorkflowIDReusePolicy

from src.workflows.training_workflow import LLMTrainingWorkflow, TrainingParams


async def start_training(args):
    """Start a new training workflow."""
    print(f"Connecting to Temporal server at {args.server}...")
    client = await Client.connect(args.server)
    
    # Parse phases
    phases = args.phases.split(",") if args.phases else ["1", "2", "2b"]
    
    # Build parameters
    params = TrainingParams(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        db_path=args.db_path,
        phases=phases,
        skip_download=args.skip_download,
        skip_clean=args.skip_clean,
        skip_tokenizer=args.skip_tokenizer,
        skip_tokenization=args.skip_tokenization,
        vocab_size=args.vocab_size,
    )
    
    # Generate workflow ID
    if args.workflow_id:
        workflow_id = args.workflow_id
    else:
        workflow_id = "llm-training-main"
    
    print(f"\nStarting workflow: {workflow_id}")
    print(f"Phases: {phases}")
    print(f"Data directory: {args.data_dir}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print("-" * 60)
    
    # Start workflow
    handle = await client.start_workflow(
        LLMTrainingWorkflow.run,
        params,
        id=workflow_id,
        task_queue="llm-training",
        id_reuse_policy=WorkflowIDReusePolicy.TERMINATE_IF_RUNNING,
    )
    
    print(f"\nWorkflow started successfully!")
    print(f"  ID: {handle.id}")
    print(f"  Run ID: {handle.result_run_id}")
    print("")
    print("Monitor progress:")
    print(f"  Web UI: http://localhost:8080/namespaces/default/workflows/{workflow_id}")
    print(f"  SQLite: sqlite3 {args.db_path} 'SELECT * FROM training_phases'")
    print("")
    print("Make sure worker is running:")
    print("  python scripts/temporal_worker.py")
    
    return workflow_id


async def check_status(args):
    """Check status of a workflow."""
    client = await Client.connect(args.server)
    
    try:
        handle = client.get_workflow_handle(args.workflow_id)
        desc = await handle.describe()
        
        print(f"Workflow: {args.workflow_id}")
        print(f"Status: {desc.status.name}")
        print(f"Started: {desc.start_time}")
        
        if desc.status.name == "COMPLETED":
            result = await handle.result()
            print(f"\nResult:")
            print(f"  Completed phases: {result.completed_phases}")
            print(f"  Total steps: {result.total_steps}")
            print(f"  Final loss: {result.final_loss}")
            print(f"  Checkpoint: {result.final_checkpoint}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


async def cancel_workflow(args):
    """Cancel a running workflow."""
    client = await Client.connect(args.server)
    
    try:
        handle = client.get_workflow_handle(args.workflow_id)
        await handle.cancel()
        print(f"Cancelled workflow: {args.workflow_id}")
    except Exception as e:
        print(f"Error cancelling workflow: {e}")
        return 1
    
    return 0


async def list_workflows(args):
    """List recent workflows."""
    client = await Client.connect(args.server)
    
    print("Recent training workflows:")
    print("-" * 60)
    
    async for workflow in client.list_workflows(
        query="WorkflowType = 'LLMTrainingWorkflow'",
    ):
        print(f"  {workflow.id}: {workflow.status.name}")


def main():
    parser = argparse.ArgumentParser(
        description="LLM Training Workflow Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--server",
        type=str,
        default="localhost:7233",
        help="Temporal server address"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Start command
    start = subparsers.add_parser("start", help="Start training workflow")
    start.add_argument("--phases", help="Comma-separated phases: 1,2,2b")
    start.add_argument("--data-dir", default="data", help="Data directory")
    start.add_argument("--checkpoint-dir", default="checkpoints", help="Checkpoint directory")
    start.add_argument("--log-dir", default="logs", help="Log directory")
    start.add_argument("--db-path", default="data/training_state.db", help="SQLite database path")
    start.add_argument("--workflow-id", help="Custom workflow ID")
    start.add_argument("--vocab-size", type=int, default=32000, help="Tokenizer vocab size")
    start.add_argument("--skip-download", action="store_true", help="Skip download step")
    start.add_argument("--skip-clean", action="store_true", help="Skip cleaning step")
    start.add_argument("--skip-tokenizer", action="store_true", help="Skip tokenizer training")
    start.add_argument("--skip-tokenization", action="store_true", help="Skip data tokenization")
    
    # Status command
    status = subparsers.add_parser("status", help="Check workflow status")
    status.add_argument("workflow_id", nargs="?", default="llm-training-main", help="Workflow ID to check")
    
    # Cancel command
    cancel = subparsers.add_parser("cancel", help="Cancel a workflow")
    cancel.add_argument("workflow_id", nargs="?", default="llm-training-main", help="Workflow ID to cancel")
    
    # List command
    list_cmd = subparsers.add_parser("list", help="List recent workflows")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == "start":
        asyncio.run(start_training(args))
    elif args.command == "status":
        asyncio.run(check_status(args))
    elif args.command == "cancel":
        asyncio.run(cancel_workflow(args))
    elif args.command == "list":
        asyncio.run(list_workflows(args))


if __name__ == "__main__":
    main()
