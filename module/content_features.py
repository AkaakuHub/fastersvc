import torch.nn.functional as F


def select_hubert_content(hidden_states, layer=9):
    if len(hidden_states) <= layer:
        raise ValueError(f"HuBERT output does not contain layer {layer}")
    return hidden_states[layer].transpose(1, 2)


def align_content_frames(features, frame_count):
    return F.interpolate(features, size=frame_count, mode="linear", align_corners=False)
