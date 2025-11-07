import argparse
import numpy as np
import os
import torch
import wandb
from datetime import datetime
from functools import partial

from .layers import TransformerLM, cross_entropy
from .optimizers import SGD, AdamW, lr_cosine_schedule
from .data import random_sample, det_sample
from .timer import Timer


def train_parser(
    parser: argparse.ArgumentParser,
    defaults: dict,
    argv,
):
    for key, value in defaults.items():
        if key in ["total_tokens"]:
            parser.add_argument(f"--{key}", dest=key, default=value, type=int)
        elif key in ["lr"]:
            parser.add_argument(f"--{key}", dest=key, default=value, type=float)
        else:
            parser.add_argument(f"--{key}", dest=key, default=value)
    return parser.parse_args(argv)


def train(data, model, params, device):
    # Random sample a batch for training.
    batch, label = random_sample(data, params.batch_size, params.context_length, device)
    logits = model(batch)
    loss = cross_entropy(logits, label)
    return loss


def validation(data, model, params, device):
    # Iterate through the eval data for validation.
    batch_size, context_length = params.batch_size, params.context_length
    total_loss, num_instances = 0, 0
    for idx in range(0, data.size, batch_size * context_length):
        batch, label = det_sample(data, idx, batch_size, context_length, device)
        logits = model(batch)
        loss = cross_entropy(logits, label)
        total_loss += loss * label.shape[0]
        num_instances += label.shape[0]
    return total_loss / num_instances


def run(argv=None):
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    config = dict(
        train_data="../data/TinyStoriesV2-GPT4-valid-ids.bin", # NOTE: change it to train ids.
        validation_data="../data/TinyStoriesV2-GPT4-valid-ids.bin",
        batch_size=4,
        vocab_size=1000,
        total_tokens=1_024_000,
        context_length=256,
        d_model=64,
        num_layers=2,
        num_heads=2,
        d_ff=256,
        use_rope=True,
        rope_theta=10_000,
        lr=1e-2,
        betas=(0.9,0.95),
        eps=1e-8,
        weight_decay=1e-5,
        alpha_min=1e-6, 
        t_w=1000, 
        t_c=19_000,
        ckpt_path="./ckpts/my_model.pt",
    )
    config = train_parser(parser, defaults=config, argv=argv)
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
    # opt = SGD(model.parameters(), lr=config.lr)
    opt = AdamW(
        model.parameters(), 
        lr=config.lr, 
        betas=config.betas, 
        eps=config.eps, 
        weight_decay=config.weight_decay, 
        lr_schedule=partial(
            lr_cosine_schedule,
            alpha_max=config.lr,
            alpha_min=config.alpha_min,
            t_w=config.t_w,
            t_c=config.t_c,
        ),
    )
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_dataset = np.memmap(os.path.join(current_dir, config.train_data), dtype=np.int32, mode='r')
    validation_dataset = np.memmap(os.path.join(current_dir, config.validation_data), dtype=np.int32, mode='r')
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    wandb.init(project="cs336", name=f"{current_time}-assignment1-model-training", config=config)

    # Training Loops.
    n_steps = config.total_tokens // config.batch_size // config.context_length
    for i in range(n_steps):
        with Timer("Training Loop") as t:
            opt.zero_grad()
            loss = train(train_dataset, model, config, device)
            loss.backward()
            opt.step()
            if i > 0 and i % 10 == 0:
                print(f"Step {i} Training Loss: {loss}")
        wandb.log({
            "batch_loss": loss.item(),
            "batch_idx": i,
            "step_time": t.elapsed,
        })
        if i > 0 and i % 100 == 0:    
            valid_loss = validation(validation_dataset[:10000], model, config, device)
            print(f"Step {i} Validation Loss: {valid_loss}")
            wandb.log({
                "train_validation_loss": valid_loss.item(),
                "batch_idx": i,
            })
    valid_loss = validation(validation_dataset[:10000], model, config, device)
    print(f"Final Validation Loss: {valid_loss}")
    wandb.log({
        "eval_validation_loss": valid_loss.item(),
        "batch_idx": i,
    })
    torch.save(model, config.ckpt_path)

if __name__ == "__main__":
    run()