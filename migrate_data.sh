#!/bin/bash
set -e

DEST="/Users/hrishikesh/.mounty/MY EXTERNAL HDD/custom-llm-data"
LOGfile="migration.log"

log() {
    echo "[$(date)] $1" | tee -a "$LOGfile"
}

log "Starting migration to $DEST"

# 1. Move mirror
if [ -d "data/mirror" ] && [ ! -L "data/mirror" ]; then
    log "Moving data/mirror (22GB)..."
    mv data/mirror "$DEST/"
    log "Linking data/mirror..."
    ln -s "$DEST/mirror" data/mirror
else
    log "data/mirror already processed or missing."
fi

# 2. Move raw
if [ -d "data/raw" ] && [ ! -L "data/raw" ]; then
    log "Moving data/raw (61GB)..."
    mv data/raw "$DEST/"
    log "Linking data/raw..."
    ln -s "$DEST/raw" data/raw
else
    log "data/raw already processed or missing."
fi

log "Migration completed successfully."
