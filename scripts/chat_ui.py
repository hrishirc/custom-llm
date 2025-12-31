#!/usr/bin/env python3
"""Gradio Chat UI for the trained model."""
import sys
import torch
import gradio as gr
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model.llm import LLM
from src.tokenizer.tokenizer import TokenizerWrapper
from src.model.config import TrainingConfig

# Global state
MODEL = None
TOKENIZER = None
DEVICE = "cpu"

def load_latest_checkpoint(checkpoint_dir: Path):
    """Find the latest checkpoint."""
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        return None
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(f"Loading checkpoint: {latest.name}")
    return latest

def initialize_model():
    """Load model and tokenizer once."""
    global MODEL, TOKENIZER, DEVICE
    
    if MODEL is not None:
        return
        
    root_dir = Path(__file__).parent.parent
    checkpoint_dir = root_dir / "checkpoints"
    tokenizer_path = root_dir / "data" / "tokenizer.json"
    
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    
    TOKENIZER = TokenizerWrapper(str(tokenizer_path))
    
    checkpoint_path = load_latest_checkpoint(checkpoint_dir)
    if not checkpoint_path:
        raise RuntimeError("No checkpoints found!")
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    # Checkpoint contains TrainingConfig, but we need ModelConfig for the model
    # We use default ModelConfig since architecture is fixed
    from src.model.config import ModelConfig
    model_config = ModelConfig()
    
    MODEL = LLM(model_config)
    MODEL.load_state_dict(checkpoint["model_state_dict"])
    MODEL.to(DEVICE)
    MODEL.eval()

def chat_response(message, history, max_tokens, temp, top_k):
    """Generate response for Gradio."""
    initialize_model()
    
    # Cast sliders to int
    max_tokens = int(max_tokens)
    top_k = int(top_k)
    
    # Simple history concatenation? 
    # For now, just use the latest message as prompt
    prompt = message
    
    # Encode
    if prompt is None:
        prompt = ""
    else:
        # Handle Gradio Multimodal format (list of dicts)
        if isinstance(prompt, list):
             text_parts = [item["text"] for item in prompt if isinstance(item, dict) and item.get("type") == "text"]
             prompt = " ".join(text_parts)
        
        prompt = str(prompt)
        
    print(f"Generating from prompt: '{prompt}' (type: {type(prompt)})")
    
    tokens = TOKENIZER.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
    
    # Generate
    generated_text = ""
    with torch.no_grad():
        for _ in range(max_tokens):
            if input_ids.size(1) > MODEL.config.max_seq_len:
                 input_ids = input_ids[:, -MODEL.config.max_seq_len:]
                 
            logits = MODEL(input_ids)["logits"]
            next_token_logits = logits[:, -1, :]
            
            if temp > 0:
                probs = torch.softmax(next_token_logits / temp, dim=-1)
                if top_k > 0:
                    v, _ = torch.topk(probs, min(top_k, probs.size(-1)))
                    probs[probs < v[:, [-1]]] = 0
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            token_id = next_token.item()
            
            if token_id == TOKENIZER.eos_token_id:
                break
            
            # Simple decoding with manual replacement for Metaspace (U+2581)
            # This avoids decode() stripping leading spaces
            word = TOKENIZER.tokenizer.id_to_token(token_id)
            if word:
                word = word.replace("\u2581", " ")
                if word in ["<BOS>", "<EOS>", "<PAD>", "<UNK>"]:
                    word = ""
                
                generated_text += word
            
            # Yield full response for streaming
            yield generated_text


def user(message, history):
    return "", history + [{"role": "user", "content": message}]

def bot(history, max_tokens, temp, top_k):
    # History is list of dicts: [{"role": "user", "content": "hi"}, ...]
    message = history[-1]["content"]
    
    # Add empty assistant message
    history.append({"role": "assistant", "content": ""})
    
    # Generate response
    # We pass history excluding the last empty assistant message
    # But our generation logic currently just uses the prompt 'message'
    
    for chunk in chat_response(message, history, max_tokens, temp, top_k):
        # Update the last message (assistant)
        history[-1]["content"] = chunk
        yield history

with gr.Blocks(title="Custom LLM Chat") as demo:
    gr.Markdown("# Custom LLM Chat (Training in Progress)")
    
    chatbot = gr.Chatbot(height=600)
    msg = gr.Textbox(placeholder="Type prompt here (e.g. 'The universe is')...", show_label=False)
    
    with gr.Accordion("Parameters", open=False):
        max_tokens = gr.Slider(10, 500, value=100, label="Max New Tokens")
        temp = gr.Slider(0.1, 2.0, value=0.7, label="Temperature")
        top_k = gr.Slider(1, 100, value=50, label="Top-K")
    
    clear = gr.Button("Clear")

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [chatbot, max_tokens, temp, top_k], chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(share=True)
