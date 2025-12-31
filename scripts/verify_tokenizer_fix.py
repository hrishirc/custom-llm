
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.tokenizer.tokenizer import create_tokenizer, train_tokenizer

# Create config
tokenizer = create_tokenizer(vocab_size=100)

# Dummy data
data = ["The universe is huge.", "The sun is bright."]

# Train
trained = train_tokenizer(tokenizer, iterator=data, vocab_size=100)

# Encode
encoded = trained.encode("The universe")
print(f"Encoded 'The universe': {encoded.tokens}")

# Decode
decoded = trained.decode(encoded.ids)
print(f"Decoded: '{decoded}'")

# Check if spaces are preserved correctly (Metaspace uses U+2581 ' ')
# We expect tokens like " The" (starting with U+2581)
has_space_char = any(" " in t for t in encoded.tokens)
if has_space_char:
    print("SUCCESS: Tokens contain space character.")
else:
    print("FAILURE: Tokens do NOT contain space character.")
