
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.tokenizer.tokenizer import TokenizerWrapper

t = TokenizerWrapper("data/tokenizer.json")
vocab = t.tokenizer.get_vocab()
# Invert map
id_to_token = {v: k for k, v in vocab.items()}

# Check for "the" vs "▁the" (U+2581)
meta = " " # U+2581

check_words = ["the", "is", "a", "universe", "s"]

print("Checking presence of space/no-space variants:")
for w in check_words:
    with_space = f"{meta}{w}"
    no_space = w
    
    id_with = vocab.get(with_space)
    id_no = vocab.get(no_space)
    
    print(f"'{w}':")
    print(f"  With space ('{with_space}'): {id_with}")
    print(f"  No space   ('{no_space}'): {id_no}")

# Also just count how many tokens start with meta char
meta_count = sum(1 for k in vocab.keys() if k.startswith(meta))
print(f"\nTokens starting with U+2581: {meta_count} / {len(vocab)}")
