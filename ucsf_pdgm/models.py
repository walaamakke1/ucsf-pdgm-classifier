import torch
from transformers import ViTModel, ViTImageProcessor


def get_model(model_id):
 
    processor = ViTImageProcessor.from_pretrained(model_id)
    model = ViTModel.from_pretrained(model_id)
    for param in model.parameters():
        param.requires_grad = False
    model.processor = processor  # attached so get_features(model, layer, X) is self-contained
    return model, processor


def get_features(model, layer, X):

    if not isinstance(X, list):
        X = [X]

    device = next(model.parameters()).device
    processor = model.processor

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