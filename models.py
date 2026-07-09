import torch
from transformers import ViTModel, ViTImageProcessor


def get_model(model_id):
    """Loads a pretrained model + matching processor, and freezes the model."""
    processor = ViTImageProcessor.from_pretrained(model_id)
    model = ViTModel.from_pretrained(model_id)
    for param in model.parameters():
        param.requires_grad = False
    return model, processor


def get_features(model, processor, layer, X, device="cpu"):
    """
    Extracts features from a given layer for input X.

    X: a single PIL image, or a list of PIL images (e.g. one volume's slices).
    layer: which output to use — currently supports "cls" (CLS token).
    Returns: a single feature vector (mean-pooled across all images in X if a list).
    """
    if not isinstance(X, list):
        X = [X]

    model = model.to(device)
    tokens = []
    for image in X:
        image = image.convert("RGB")
        inputs = processor(image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)

        if layer == "cls":
            token = outputs.last_hidden_state[:, 0, :]
        else:
            raise ValueError(f"Unsupported layer: {layer}")

        tokens.append(token)

    return torch.mean(torch.stack(tokens), dim=0).squeeze(0)