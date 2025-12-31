
import os
import time
import shutil

SOURCE_DIRS = ["data/mirror", "data/raw"]
DEST_BASE = "/Users/hrishikesh/.mounty/MY EXTERNAL HDD/custom-llm-data"
LOG_FILE = "migration_gentle.log"
CHUNK_SIZE = 1024 * 1024  # 1MB
CHUNK_DELAY = 0.05        # 50ms pause between chunks -> max ~20MB/s theoretical, likely lower

def log(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def copy_file_gentle(src, dst):
    """Copy file in chunks with delays to avoid overwhelming NTFS driver."""
    try:
        # Ensure dir exists
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        
        with open(src, 'rb') as fsrc:
            with open(dst, 'wb') as fdst:
                while True:
                    chunk = fsrc.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    fdst.write(chunk)
                    # Flush to force write to OS buffer, but maybe too aggressive?
                    # fdst.flush() 
                    # os.fsync(fdst.fileno()) # Very slow, but safe. Let's try just sleep first.
                    time.sleep(CHUNK_DELAY)
        
        # Try to copy metadata/times
        try:
            shutil.copystat(src, dst)
        except:
            pass
            
        return True
    except Exception as e:
        log(f"Error copying {src}: {e}")
        return False

def copy_file_with_retry(src, dst, max_retries=5):
    retries = 0
    while retries < max_retries:
        if copy_file_gentle(src, dst):
            return True
        
        retries += 1
        log(f"Retry {retries}/{max_retries} for {os.path.basename(src)} in 5s...")
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
            
            # Check if already done (size match)
            if os.path.exists(dst_file):
                if os.path.getsize(src_file) == os.path.getsize(dst_file):
                    try:
                        os.remove(src_file)
                    except OSError:
                        pass
                    continue
            
            # Copy
            success = copy_file_with_retry(src_file, dst_file)
            if success:
                # Verify and delete
                verify_and_delete(src_file, dst_file)
            else:
                log(f"FAILED to migrate {src_file} after retries. SKIPPING to proceed with others.")
                # Don't return, just continue to next file
                continue

    # Cleanup empty directories
    for root, dirs, files in os.walk(src_root, topdown=False):
        for name in dirs:
            d = os.path.join(root, name)
            try:
                os.rmdir(d)
            except OSError:
                pass 
    
    # Symlink if fully empty
    if not os.listdir(src_root):
        os.rmdir(src_root)
        os.symlink(dst_root, src_root)
        log(f"Successfully symlinked {src_root} -> {dst_root}")
    else:
        log(f"Finished {folder_name} but source dir not empty (failures occurred).")

if __name__ == "__main__":
    log("Starting GENLTE Python Migration...")
    try:
        migrate_folder("data/mirror")
        migrate_folder("data/raw")
        log("Migration finished.")
    except KeyboardInterrupt:
        log("Migration interrupted by user.")
    except Exception as e:
        log(f"Critical script error: {e}")
