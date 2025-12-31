
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.tokenizer.tokenizer import TokenizerWrapper

t = TokenizerWrapper("data/tokenizer.json")

# ID 18 is '▁is'
tid = 18
decoded = t.tokenizer.decode([tid])
print(f"ID {tid}: Decoded='{decoded}' (len={len(decoded)})")

# Check if spaces are stripped
# '▁is' should become ' is'
if decoded == "is":
    print("WARNING: Leading space was STRIPPED.")
elif decoded == " is":
    print("SUCCESS: Leading space preserved.")
else:
    print(f"Unknown result: '{decoded}'")

# Try to prevent stripping
# HuggingFace tokenizers decode usually strips?
# There isn't always a parameter for it in `decode([ids])`.
# But `id_to_token` gives raw token.
raw = t.tokenizer.id_to_token(tid)
print(f"Raw: '{raw}'")
manual = raw.replace("\u2581", " ")
print(f"Manual Replace: '{manual}'")
if manual == " is":
    print("SUCCESS: Manual replacement preserved space.")
