import numpy as np
import torch


def random_sample(dataset, batch_size, context_length, device="cpu"):
    # Sample starting points.
    start = np.random.choice(dataset.size - context_length, batch_size, replace=True)
    data = []
    label = []
    for i in start:
        data.append(dataset[i:i+context_length])
        label.append(dataset[i+1:i+context_length+1])
    data = torch.tensor(
        np.stack(data, axis=0),
        dtype=torch.long,
        device=device,
    )
    label = torch.tensor(
        np.stack(label, axis=0),
        dtype=torch.long,
        device=device,
    )
    return data, label


def det_sample(dataset, start, batch_size, context_length, device="cpu"):
    data = []
    label = []
    offset = batch_size * context_length
    for i in range(start, start+offset, context_length):
        if i+context_length >= dataset.size:
            break
        data.append(dataset[i:i+context_length])
        label.append(dataset[i+1:i+context_length+1])
    data = torch.tensor(
        np.stack(data, axis=0),
        dtype=torch.long,
        device=device,
    )
    label = torch.tensor(
        np.stack(label, axis=0),
        dtype=torch.long,
        device=device,
    )
    return data, label