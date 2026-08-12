import torch


def discriminator_loss(real_logits, generated_logits):
    if len(real_logits) != len(generated_logits):
        raise ValueError("real and generated discriminator outputs must have equal lengths")
    losses = [
        ((real - 1) ** 2).mean() + (generated ** 2).mean()
        for real, generated in zip(real_logits, generated_logits)
    ]
    return torch.stack(losses).mean()


def generator_adversarial_loss(generated_logits):
    losses = [((generated - 1) ** 2).mean() for generated in generated_logits]
    return torch.stack(losses).mean()


def require_finite(name, tensor):
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name} contains a non-finite value")
