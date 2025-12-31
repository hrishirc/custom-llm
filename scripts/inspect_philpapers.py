
import sys
from datasets import load_dataset

try:
    print("Loading stanford_plato...")
    ds = load_dataset("hugfaceguy0001/stanford_plato", split="train", streaming=True)
    
    print("Getting first sample...")
    sample = next(iter(ds))
    
    print("Keys:", sample.keys())
    print("main_text type:", type(sample.get("main_text")))
    print("main_text preview:", str(sample.get("main_text"))[:200])
    
except Exception as e:
    print(f"Error: {e}")
