from tqdm.auto import tqdm

from torch.utils.data import DataLoader

from torch.optim import Optimizer

from torch.optim.lr_scheduler import LRScheduler

from torch.nn import CrossEntropyLoss

from . import lstm_model


def train_one_epoch(
        model: lstm_model.LSTMAutoComplete, 
        device: str, 
        epoch: int, 
        train_dataloader: DataLoader, 
        optimizer: Optimizer, 
        criterion: CrossEntropyLoss, 
        scheduler:LRScheduler=None):
    
    model.train()
    train_loss = 0.
    for batch in tqdm(train_dataloader, desc=f'Training epoch {epoch}'):
        source = batch['source'].to(device)
        target = batch['target'].to(device)

        logits = model(source)
        loss = criterion(logits.permute(0, 2, 1), target)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    train_loss /= len(train_dataloader)
    print(f'Epoch: {epoch}, training loss: {train_loss:.4f}')

    return train_loss