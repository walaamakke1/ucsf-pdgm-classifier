import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ucsf_pdgm.datasets import get_dataset
from ucsf_pdgm.models import get_model, get_features


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels, label_to_idx):
        self.embeddings = embeddings
        self.labels = [label_to_idx[l] for l in labels]

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def downstream(model_id, dataset_id):

    epochs, lr, batch_size = 30, 1e-3, 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "checkpoint.pt"

    train_v = get_dataset(dataset_id, "train")
    val_v = get_dataset(dataset_id, "val")
    test_v = get_dataset(dataset_id, "test")

    model, _ = get_model(model_id)
    model = model.to(device)

    def embed(vids):
        embs, labs = [], []
        for v in vids:
            embs.append(get_features(model, "cls", v["slices"]))
            labs.append(v["label"] if v["label"] else "")
        return embs, labs

    train_e, train_l = embed(train_v)
    val_e, val_l = embed(val_v)
    test_e, test_l = embed(test_v)

    unique_labels = sorted(set(train_l + val_l + test_l))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}

    train_loader = DataLoader(EmbeddingDataset(train_e, train_l, label_to_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(EmbeddingDataset(val_e, val_l, label_to_idx), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(EmbeddingDataset(test_e, test_l, label_to_idx), batch_size=batch_size, shuffle=False)

    classifier = nn.Linear(model.config.hidden_size, len(unique_labels)).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    def evaluate(loader):
        classifier.eval()
        correct, total, total_loss = 0, 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = classifier(x)
                total_loss += loss_fn(logits, y).item() * y.size(0)
                correct += (torch.argmax(logits, dim=1) == y).sum().item()
                total += y.size(0)
        classifier.train()
        return total_loss / total, correct / total

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_state, best_epoch = 0, None, 0

    for epoch in range(epochs):
        classifier.train()
        total_loss, correct, total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = classifier(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (torch.argmax(logits, dim=1) == y).sum().item()
            total += y.size(0)

        train_loss, train_acc = total_loss / total, correct / total
        val_loss, val_acc = evaluate(val_loader)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc, best_state, best_epoch = val_acc, classifier.state_dict(), epoch

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f} Acc={val_acc:.4f}")

    classifier.load_state_dict(best_state)
    test_loss, test_acc = evaluate(test_loader)
    history["test_loss"] = test_loss
    history["test_acc"] = test_acc

    checkpoint = {
        "state_dict": best_state,
        "optimizer": optimizer.state_dict(),
        "last_epoch": best_epoch,
        "history": history,
        "label_to_idx": label_to_idx,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"\nSaved checkpoint to {checkpoint_path}")
    print(f"Best epoch: {best_epoch + 1} | Val Acc: {best_val_acc:.4f} | Test Acc: {test_acc:.4f}")
    return checkpoint