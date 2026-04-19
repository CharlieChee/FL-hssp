"""Generic CNN training / evaluation loops used by ``cnn.py --mode train``."""
import torch


def train_epoch(model, device, loader, optimizer, criterion, epoch=None, log_interval=10):
    """Run one training epoch and return ``(mean_loss, accuracy_percent)``."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    n_batches = len(loader)
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        total_loss += loss_val
        pred = out.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        if log_interval > 0 and (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            acc = 100.0 * correct / total
            ep_str = f"Epoch {epoch} " if epoch is not None else ""
            print(
                f"  {ep_str}Batch {batch_idx + 1}/{n_batches}  "
                f"loss={loss_val:.4f}  avg_loss={avg_loss:.4f}  "
                f"acc={acc:.2f}%  ({correct}/{total})"
            )

    return total_loss / n_batches, 100.0 * correct / total


def evaluate(model, device, loader, criterion):
    """Compute ``(mean_loss, accuracy_percent)`` on ``loader`` under ``torch.no_grad``."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss = criterion(out, target)
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return total_loss / len(loader), 100.0 * correct / total
