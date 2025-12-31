#!/bin/bash
# Migration script with Auto-Retry
set -e

DEST="/Users/hrishikesh/.mounty/MY EXTERNAL HDD/custom-llm-data"
LOGfile="migration_rsync.log"

log() {
    echo "[$(date)] $1" | tee -a "$LOGfile"
}

run_rsync_with_retry() {
    local src="$1"
    local dest_parent="$2"
    local max_retries=10
    local count=0
    local delay=30

    while [ $count -lt $max_retries ]; do
        log "Syncing $src to $dest_parent (Attempt $((count+1))/$max_retries)..."
        
        # Try to ensure destination exists (drive might have just been remounted)
        if [ ! -d "$dest_parent" ]; then
             log "Destination $dest_parent not found. Attempting mkdir..."
             # We use || true here to suppress exit on failure, trusting rsync to report the error if it persists
             mkdir -p "$dest_parent" || log "mkdir failed (drive missing?)"
        fi

        # Run rsync
        set +e
        rsync -av --remove-source-files --progress "$src" "$dest_parent" >> "$LOGfile" 2>&1
        local exit_code=$?
        set -e

        if [ $exit_code -eq 0 ]; then
            log "Success: $src synced."
            return 0
        else
            log "Error: rsync failed with exit code $exit_code."
            count=$((count + 1))
            if [ $count -lt $max_retries ]; then
                log "Retrying in $delay seconds... (Please check drive connection)"
                sleep $delay
            else
                log "Failed: Max retries reached for $src."
                return 1
            fi
        fi
    done
}

log "Starting AUTO-RETRY migration to $DEST"

# 1. Sync mirror
if [ -d "data/mirror" ] && [ ! -L "data/mirror" ]; then
    run_rsync_with_retry "data/mirror" "$DEST/"
    
    # Remove empty dirs (rsync --remove-source-files leaves empty dirs behind)
    find data/mirror -type d -empty -delete
    
    # Check if directory is effectively gone (or only symlink candidate remains)
    if [ -d "data/mirror" ] && [ "$(ls -A data/mirror)" ]; then
        log "WARNING: data/mirror is not empty after sync. Checking content..."
        ls -F data/mirror | tee -a "$LOGfile"
    else
        log "Linking data/mirror..."
        rm -rf data/mirror
        ln -s "$DEST/mirror" data/mirror
    fi
else
    log "data/mirror already processed or missing."
fi

# 2. Sync raw
if [ -d "data/raw" ] && [ ! -L "data/raw" ]; then
    run_rsync_with_retry "data/raw" "$DEST/"
    
    find data/raw -type d -empty -delete
    
    if [ -d "data/raw" ] && [ "$(ls -A data/raw)" ]; then
        log "WARNING: data/raw is not empty after sync. Checking content..."
        ls -F data/raw | tee -a "$LOGfile"
    else
        log "Linking data/raw..."
        rm -rf data/raw
        ln -s "$DEST/raw" data/raw
    fi
else
    log "data/raw already processed or missing."
fi

log "Migration completed successfully."
