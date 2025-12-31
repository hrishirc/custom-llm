
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.tokenizer.tokenizer import TokenizerWrapper

t = TokenizerWrapper("data/tokenizer.json")
print(f"Vocab size: {t.tokenizer.get_vocab_size()}")

# Check some common words
for word in [" The", "The", " universe", "universe", " is", "is"]:
    tokens = t.tokenizer.encode(word).tokens
    print(f"'{word}' -> {tokens}")

# Check first 100 tokens
vocab = t.tokenizer.get_vocab()
# Invert map
id_to_token = {v: k for k, v in vocab.items()}
print("\nFirst 20 tokens:")
for i in range(20):
    print(f"{i}: {id_to_token.get(i)}")
