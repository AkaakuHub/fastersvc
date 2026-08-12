import torch.nn as nn


def step_scaled_optimizer(loss, optimizer, scaler, parameters, max_gradient_norm):
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradient_norm = nn.utils.clip_grad_norm_(parameters, max_gradient_norm)
    scaler.step(optimizer)
    return gradient_norm
