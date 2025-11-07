import argparse
import os
import torch

from .layers import TransformerLM, softmax
from .tokenizer import Tokenizer


def decode_parser(
    parser: argparse.ArgumentParser, 
    defaults: dict,
    argv,
):
    for key, value in defaults.items():
        parser.add_argument(f"--{key}", dest=key, default=value)
    return parser.parse_args(argv)


def greedy_decode(logits: torch.Tensor):
    return torch.argmax(logits).view(1,1)


def sample_decode(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0):
    logits = logits / temperature
    probs = softmax(logits)
    if top_p < 1.0:
        sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
        cdf = torch.cumsum(sorted_probs, dim=-1)
        idx = torch.sum(cdf<top_p, dim=-1) + 1
        sorted_probs[idx:] = 0
        inv_indices = torch.argsort(indices)
        probs = sorted_probs[inv_indices] / sum(sorted_probs)
    return torch.multinomial(probs, 1).view(1,1)


def run(argv=None):
    parser = argparse.ArgumentParser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config = dict(
        prompt="prompt",
        vocab_file_path=os.path.join(current_dir, "../ckpts/TinyStoriesV2-GPT4-tokenizer-train-vocab.pkl"),
        merges_file_path=os.path.join(current_dir, "../ckpts/TinyStoriesV2-GPT4-tokenizer-train-merges.pkl"),
        vocab_size=1000,
        context_length=256,
        d_model=64,
        num_layers=2,
        num_heads=2,
        d_ff=256,
        use_rope=True,
        rope_theta=10_000,
        ckpt_path="./ckpts/my_model.pt",
        max_decode_len=238,
    )
    config = decode_parser(parser, defaults=config, argv=argv)
    tokenizer = Tokenizer.from_files(
        config.vocab_file_path,
        config.merges_file_path,
        ["<|endoftext|>"],
    )
    model = TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        use_rope=config.use_rope,
        rope_theta=config.rope_theta,
    ).to(device)
    model = torch.load(config.ckpt_path, weights_only=False).to(device)
    stop_token = [tokenizer.reverse_vocab[token.encode("utf-8")] for token in tokenizer.special_tokens]
    prompt = "Once upon a time, there was a little girl named Lily."
    prompt = torch.tensor(tokenizer.encode(prompt))[None,:].to(device)
    decode_fn = sample_decode

    # Decoding Loops.
    for _ in range(config.max_decode_len):
        logits = model(prompt)
        token = decode_fn(logits[0,-1,:], temperature=1.0, top_p=0.9)
        if token[0][0] in stop_token:
            break
        prompt = torch.cat([prompt, token], dim=-1)
    response = tokenizer.decode(prompt.data[0].tolist())
    print("LLM output: ", response)


if __name__ == "__main__":
    run()