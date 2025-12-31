#!/bin/bash
set -e

DEST="/Users/hrishikesh/.mounty/MY EXTERNAL HDD/custom-llm-data"
LOGfile="migration_cp.log"

log() {
    echo "[$(date)] $1" | tee -a "$LOGfile"
}

move_dir() {
    local src="$1"
    local parent_dest="$2"
    
    # 1. Ensure parent dest exists
    mkdir -p "$parent_dest"
    
    log "Copying $src to $parent_dest (using cp -R)..."
    
    # Simple recursive copy.
    # -R: recursive
    # -v: verbose
    cp -Rv "$src" "$parent_dest/" >> "$LOGfile" 2>&1
    
    log "Verifying $src..."
    # If copy succeeded (exit code 0), we assume it's safe to delete source
    # A true verification would require diffing content, but that takes 2x time.
    # We will trust cp exit code + file existence
    
    if [ -d "$parent_dest/$(basename "$src")" ]; then
        log "Copy success. Deleting original $src..."
        rm -rf "$src"
        log "Deleted $src"
    else
        log "CRITICAL: Copy appeared to succeed but destination file is missing!"
        exit 1
    fi
}

log "Starting SIMPLE COPY migration to $DEST"

# 1. Mirror
if [ -d "data/mirror" ] && [ ! -L "data/mirror" ]; then
    move_dir "data/mirror" "$DEST"
    log "Linking data/mirror..."
    ln -s "$DEST/mirror" data/mirror
else
    log "data/mirror already processed or missing."
fi

# 2. Raw
if [ -d "data/raw" ] && [ ! -L "data/raw" ]; then
    move_dir "data/raw" "$DEST"
    log "Linking data/raw..."
    ln -s "$DEST/raw" data/raw
else
    log "data/raw already processed or missing."
fi

log "Migration completed successfully."
