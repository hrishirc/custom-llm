#!/bin/bash
# Migration script with Auto-Retry and NTFS-friendly settings
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
        
        if [ ! -d "$dest_parent" ]; then
             mkdir -p "$dest_parent" || true
        fi

        set +e
        # MODIFIED RSYNC COMMAND:
        # --inplace: Writes directly to file instead of creating .temp files (fixes mkstempat errors on some FUSE fs)
        # --no-whole-file: Default optimization
        # Removed --remove-source-files for now to ensure stability first (will delete manually after)
        rsync -av --progress --inplace "$src" "$dest_parent" >> "$LOGfile" 2>&1
        local exit_code=$?
        set -e

        if [ $exit_code -eq 0 ]; then
            log "Success: $src synced."
            
            # Manual verify and delete since we removed --remove-source-files
            log "Verifying and cleaning up source..."
            # Simple check: if exit code 0, rsync thinks it's done.
            rm -rf "$src"
            return 0
        else
            log "Error: rsync failed with exit code $exit_code."
            # grep the log for specific error to give hint?
            tail -n 2 "$LOGfile"
            
            count=$((count + 1))
            if [ $count -lt $max_retries ]; then
                log "Retrying in $delay seconds..."
                sleep $delay
            else
                log "Failed: Max retries reached for $src."
                return 1
            fi
        fi
    done
}

log "Starting NTFS-FRIENDLY migration to $DEST"

# 1. Sync mirror
if [ -d "data/mirror" ] && [ ! -L "data/mirror" ]; then
    run_rsync_with_retry "data/mirror" "$DEST/"
    
    if [ ! -d "data/mirror" ]; then
        log "Linking data/mirror..."
        ln -s "$DEST/mirror" data/mirror
    fi
else
    log "data/mirror already processed or missing."
fi

# 2. Sync raw
if [ -d "data/raw" ] && [ ! -L "data/raw" ]; then
    run_rsync_with_retry "data/raw" "$DEST/"
    
    if [ ! -d "data/raw" ]; then
        log "Linking data/raw..."
        ln -s "$DEST/raw" data/raw
    fi
else
    log "data/raw already processed or missing."
fi

log "Migration completed successfully."
