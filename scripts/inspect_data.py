
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.tokenizer.tokenizer import TokenizerWrapper

# Load data
data_path = "data/tokenized/phase1.npy"
if not Path(data_path).exists():
    print(f"File not found: {data_path}")
    sys.exit(1)

ids = np.memmap(data_path, dtype=np.int32, mode="r")
sample_ids = ids[:50]

print(f"Sample IDs: {sample_ids}")

# Load tokenizer to decode
t = TokenizerWrapper("data/tokenizer.json")

print("\nDecoding individual IDs:")
for i, tid in enumerate(sample_ids):
    token = t.tokenizer.decode([tid])
    # raw token string from vocab
    raw_token = t.tokenizer.id_to_token(tid)
    print(f"{i}: ID {tid} -> '{raw_token}' -> Decoded: '{token}'")

print("\nFull Decode:")
full_text = t.decode(sample_ids)
print(f"'{full_text}'")
