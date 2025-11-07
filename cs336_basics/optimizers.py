from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss
    

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay, lr_schedule=lambda x:x):
        defaults = {
            "alpha": lr,
            "beta1": betas[0],
            "beta2": betas[1],
            "eps": eps,
            "lambda_": weight_decay,
        }
        self.lr_schedule=lr_schedule
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            alpha = group["alpha"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            lambda_ = group["lambda_"]
            for p in group["params"]:
                # States associated with p.
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                # Rate should follow lr schedule.
                alpha = self.lr_schedule(t)
                # Update states with grads.
                g = p.grad.data
                upt_m = beta1 * m + (1 - beta1) * g
                upt_v = beta2 * v + (1 - beta2) * torch.square(g)
                alpha_t = alpha * math.sqrt(1 - pow(beta2, t))/(1 - pow(beta1, t))
                p.data -= alpha_t * upt_m / (torch.sqrt(upt_v) + eps)
                p.data *= (1 - alpha*lambda_)
                state["m"] = upt_m
                state["v"] = upt_v
                state["t"] = t + 1
        return loss
    

def lr_cosine_schedule(t: int, alpha_max: float, alpha_min: float, t_w: int, t_c: int):
    if t < t_w:
        return t / t_w * alpha_max
    elif t >= t_w and t <= t_c:
        return alpha_min + 0.5 * (1 + math.cos((t-t_w)/(t_c-t_w)*math.pi)) * (alpha_max - alpha_min)
    else:
        return alpha_min
    

def gradient_clipping(parameters, max_l2_norm, eps=1e-6):
    global_norm = 0
    for param in parameters:
        if param.grad is not None:
            global_norm += torch.sum(torch.square(param.grad.data))
    global_norm = torch.sqrt(global_norm)
    if global_norm >= max_l2_norm:
        for param in parameters:
            if param.grad is not None:
                param.grad.data *= max_l2_norm / (global_norm + eps)
    return parameters


# weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
# opt = SGD([weights], lr=1)

# for t in range(100):
#     opt.zero_grad() # Reset the gradients for all learnable parameters.
#     loss = (weights**2).mean() # Compute a scalar loss value.
#     print(loss.cpu().item())
#     loss.backward() # Run backward pass, which computes gradients.
#     opt.step() # Run optimizer step.