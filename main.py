import torch
from datasets import load_dataset
import torch.nn as nn
from transformers import AutoTokenizer

from src.hrm import HRM, HRMConfig


# -------------------------
# Example / minimal usage
# -------------------------


def _demo_train_loop():
    """
    Tiny demo: teach the model to classify whether the sum of tokens is even (toy task).

    This demonstrates deep supervision and one-step gradient; it's not meant to be accurate.
    """
    torch.manual_seed(0)

    seq_len = 16
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab = tokenizer.vocab_size

    cfg = HRMConfig(
        vocab_size=vocab,
        d_model=256,
        n_heads=8,
        depth_L=2,
        depth_H=2,
        max_len=seq_len,
        N_cycles=2,
        T_steps=2,
        stablemax=False,
        act_enabled=False,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HRM(cfg).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    # -------------------------
    # TinyStories streaming dataset → toy parity labels
    # -------------------------
    raw_dataset = load_dataset(
        "roneneldan/TinyStories", split="train", streaming=True
    )

    text_iter = iter(raw_dataset)

    def batch(bs=32):
        texts = []
        for _ in range(bs):
            text = ""
            while text == "":
                example = next(text_iter)
                text = example.get("text", "")
            texts.append(text)
        toks = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=seq_len,
            return_tensors="pt",
        )
        ids = toks.input_ids.to(device)
        # create next-token LM pairs
        x = ids[:, :-1]
        y = ids[:, 1:]
        return x, y

    steps = 1000
    # ---- Train: always unroll to fixed segments (no ACT halting) ----
    for step in range(steps):
        x, y = batch()
        out = model(x, segments=4, teacher=y)
        loss = out["loss"]
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        print(f"step {step+1:03d} | loss {loss.item():.4f}")

    # -------------------------
    # Simple test inference using the next batch string
    # -------------------------
    model.eval()
    with torch.no_grad():
        # Use the very next batch (bs=1) as prompt
        x_prompt, _ = batch(bs=1)
        x_prompt = x_prompt.to(device)
        pad_id = tokenizer.pad_token_id
        prompt_ids = [t for t in x_prompt[0].tolist() if t != pad_id]
        input_ids = list(prompt_ids)

        max_new_tokens = 32
        for _ in range(max_new_tokens):
            x_in = torch.tensor(input_ids, device=device).unsqueeze(0)
            logits = model(x_in, act_infer_max=4)[
                "logits"]  # (1, L, V)
            next_id = int(logits[0, -1].argmax().item())
            input_ids.append(next_id)
            if next_id == tokenizer.eos_token_id:
                break

        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
        generated_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        print("\n=== Inference Sample ===")
        print(f"Prompt: {prompt_text!r}")
        print(f"Output: {generated_text!r}\n")


if __name__ == "__main__":
    _demo_train_loop()
