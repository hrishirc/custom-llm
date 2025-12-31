
import os
import shutil
import time
import hashlib

SOURCE_DIRS = ["data/mirror", "data/raw"]
DEST_BASE = "/Users/hrishikesh/.mounty/MY EXTERNAL HDD/custom-llm-data"
LOG_FILE = "migration_py.log"

def log(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def copy_file_with_retry(src, dst, max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            # Ensure dir exists
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            
            # Copy content (copy2 preserves metadata)
            shutil.copy2(src, dst)
            return True
        except Exception as e:
            retries += 1
            log(f"Error copying {src}: {e}. Retry {retries}/{max_retries} in 5s...")
            time.sleep(5)
    return False

def verify_and_delete(src, dst):
    # Simple size check
    try:
        src_size = os.path.getsize(src)
        dst_size = os.path.getsize(dst)
        if src_size == dst_size:
            os.remove(src)
            return True
        else:
            log(f"Size mismatch: {src} ({src_size}) != {dst} ({dst_size})")
            return False
    except Exception as e:
        log(f"Verification failed for {src}: {e}")
        return False

def migrate_folder(folder_name):
    src_root = os.path.abspath(folder_name)
    dst_root = os.path.join(DEST_BASE, os.path.basename(folder_name))
    
    if not os.path.exists(src_root):
        log(f"Source {src_root} does not exist. Skipping.")
        return

    if os.path.islink(src_root):
        log(f"Source {src_root} is already a symlink. Skipping.")
        return

    log(f"Migrating {src_root} -> {dst_root}")
    
    # Walk top-down
    for root, dirs, files in os.walk(src_root, topdown=True):
        for name in files:
            src_file = os.path.join(root, name)
            rel_path = os.path.relpath(src_file, start=src_root)
            dst_file = os.path.join(dst_root, rel_path)
            
            if os.path.exists(dst_file):
                # Check if already done (size match)
                if os.path.getsize(src_file) == os.path.getsize(dst_file):
                    # Already copied, just delete source
                    try:
                        os.remove(src_file)
                        # Remove empty parent dir if empty? No, do it later.
                    except OSError as e:
                        log(f"Failed to delete existing source {src_file}: {e}")
                    continue
            
            # Copy
            success = copy_file_with_retry(src_file, dst_file)
            if success:
                # Verify and delete
                verify_and_delete(src_file, dst_file)
            else:
                log(f"FAILED to migrate {src_file}. Stopping to avoid data loss.")
                return

    # Cleanup empty directories
    for root, dirs, files in os.walk(src_root, topdown=False):
        for name in dirs:
            d = os.path.join(root, name)
            try:
                os.rmdir(d)
            except OSError:
                pass # Not empty
    
    # Finally, if src_root is empty, remove it and symlink
    if not os.listdir(src_root):
        os.rmdir(src_root)
        os.symlink(dst_root, src_root)
        log(f"Successfully symlinked {src_root} -> {dst_root}")
    else:
        log(f"Finished {folder_name} but source dir not empty (failures occurred).")

if __name__ == "__main__":
    log("Starting Python Migration...")
    try:
        migrate_folder("data/mirror")
        migrate_folder("data/raw")
        log("Migration finished.")
    except KeyboardInterrupt:
        log("Migration interrupted by user.")
    except Exception as e:
        log(f"Critical script error: {e}")
