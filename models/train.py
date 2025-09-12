import torch
from torch import nn
from tqdm import tqdm
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

class FullModel(nn.Module):
    def __init__(self, pse, transformer):
        super().__init__()
        self.pse = pse
        self.transformer = transformer

    def forward(self, batch):
        pixels = batch["pixels"]           
        mask = batch["valid_pixels"]       
        extra = batch["extra"]             
        acquisition_times = batch["positions"]  


        if pixels.shape[2] < pixels.shape[3]:
            pixels = pixels.permute(0, 1, 3, 2)  
        
        x = self.pse(((pixels, mask), extra))  
        
        out = self.transformer(x, acquisition_times)
        return out


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_model_state = None
        self.verbose = verbose

    def __call__(self, val_loss, model=None):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if model is not None:
                self.best_model_state = model.state_dict()
            if self.verbose:
                print(f"New best loss: {val_loss:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f" No improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered.")

    def restore_best_weights(self, model):
        if self.best_model_state:
            model.load_state_dict(self.best_model_state)


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler=None,
    num_epochs=100,
    num_classes=8,
    save_name="best_model.pt",
    device="cuda"
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_precs, val_precs = [], []
    train_recs, val_recs = [], []
    train_f1s, val_f1s = [], []
    train_prec_w, val_prec_w = [], []
    train_rec_w, val_rec_w = [], []
    train_f1_w, val_f1_w = [], []

    early_stopping = EarlyStopping(patience=5, min_delta=0.001, verbose=True)
     # Setup torchmetrics
    acc_metric   = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    prec_metric  = Precision(task="multiclass", average="micro", num_classes=num_classes).to(device)
    rec_metric   = Recall(task="multiclass", average="micro", num_classes=num_classes).to(device)
    f1_metric    = F1Score(task="multiclass", average="micro", num_classes=num_classes).to(device)
    # weighted metrics
    prec_w_metric = Precision(task="multiclass", average="weighted", num_classes=num_classes).to(device)
    rec_w_metric  = Recall(task="multiclass", average="weighted", num_classes=num_classes).to(device)
    f1_w_metric   = F1Score(task="multiclass", average="weighted", num_classes=num_classes).to(device)

     # Training phase
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        total_samples = 0
        acc_metric.reset(); prec_metric.reset(); rec_metric.reset(); f1_metric.reset()
        prec_w_metric.reset(); rec_w_metric.reset(); f1_w_metric.reset()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            for k in batch:
                batch[k] = batch[k].to(device)

            labels = batch["label"]
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            preds = torch.argmax(outputs, dim=1)
            acc_metric.update(preds, labels)
            prec_metric.update(preds, labels)
            rec_metric.update(preds, labels)
            f1_metric.update(preds, labels)
            prec_w_metric.update(preds, labels)
            rec_w_metric.update(preds, labels)
            f1_w_metric.update(preds, labels)

        avg_train_loss = train_loss / total_samples
        train_losses.append(avg_train_loss)
        train_accs.append(acc_metric.compute().item())
        train_precs.append(prec_metric.compute().item())
        train_recs.append(rec_metric.compute().item())
        train_f1s.append(f1_metric.compute().item())
        train_prec_w.append(prec_w_metric.compute().item())
        train_rec_w.append(rec_w_metric.compute().item())
        train_f1_w.append(f1_w_metric.compute().item())


         # Validation phase
        model.eval()
        val_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                for k in batch:
                    batch[k] = batch[k].to(device)
                labels = batch["label"]
                outputs = model(batch)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                preds = torch.argmax(outputs, dim=1)
                total_samples += labels.size(0)


                reds = torch.argmax(outputs, dim=1)
                acc_metric.update(preds, labels)
                prec_metric.update(preds, labels)
                rec_metric.update(preds, labels)
                f1_metric.update(preds, labels)
                prec_w_metric.update(preds, labels)
                rec_w_metric.update(preds, labels)
                f1_w_metric.update(preds, labels)

        avg_val_loss = val_loss / total_samples
        val_losses.append(avg_val_loss)
        val_accs.append(acc_metric.compute().item())
        val_precs.append(prec_metric.compute().item())
        val_recs.append(rec_metric.compute().item())
        val_f1s.append(f1_metric.compute().item())
        val_prec_w.append(prec_w_metric.compute().item())
        val_rec_w.append(rec_w_metric.compute().item())
        val_f1_w.append(f1_w_metric.compute().item())


        print(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"Train Loss={avg_train_loss:.4f}, Acc={train_accs[-1]:.4f}, "
                f"Prec={train_precs[-1]:.4f}, Rec={train_recs[-1]:.4f}, F1={train_f1s[-1]:.4f} | "
                f"Val Loss={avg_val_loss:.4f}, Acc={val_accs[-1]:.4f}, "
                f"Prec={val_precs[-1]:.4f}, Rec={val_recs[-1]:.4f}, F1={val_f1s[-1]:.4f}"
            )

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            break

    early_stopping.restore_best_weights(model)
    torch.save(model.state_dict(), save_name)

    return (
        train_losses, val_losses,
        train_accs, val_accs,
        train_precs, val_precs,
        train_recs, val_recs,
        train_f1s, val_f1s,
        train_prec_w, val_prec_w,
        train_rec_w, val_rec_w,
        train_f1_w, val_f1_w
    )


def plot_history(
train_losses, val_losses,
train_accs, val_accs,
):
    """
    Plot training and validation metrics over epochs. Supports loss, accuracy, precision, recall, and F1-score.

    Parameters:
        train_losses, val_losses: Lists of loss values
        train_accs, val_accs:  Lists of accuracy values
        train_precs, val_precs:  Lists of precision values
        train_recs, val_recs:  Lists of recall values
        train_f1s, val_f1s: Lists of F1 score values
        train_prec_w, val_prec_w,
        train_rec_w, val_rec_w,
        train_f1_w, val_f1_w
    """
    metrics = [
        ("Loss", train_losses, val_losses),
    ]
    if train_accs is not None:
        metrics.append(("Accuracy", train_accs, val_accs))
    
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)
    for ax, (name, train_vals, val_vals) in zip(axes, metrics):
        ax.plot(epochs, train_vals, label=f"Train {name}")
        ax.plot(epochs, val_vals, label=f"Val {name}")
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.legend()

    plt.tight_layout()
    plt.show()



def evaluate_classification(model, loader, class_names):
    """
    Evaluate a classification model and print metrics and confusion matrix.

    Parameters:
        model: Trained torch.nn.Module
        loader: DataLoader yielding 
        class_names: List of class names corresponding to label indices
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            logits = model(batch)
            preds = torch.argmax(logits, dim=1)
            
            labels = batch["label"] .to(device)       
            
            all_preds.append(preds.cpu().numpy().ravel())
            all_trues.append(labels.cpu().numpy().ravel())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_trues)

    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")

    
    if isinstance(class_names, dict):
        # dict: idx → name
        num_classes  = len(class_names)
        labels       = list(range(num_classes))
        target_names = [class_names[i] for i in labels]
    else:
      
        num_classes  = len(class_names)
        labels       = list(range(num_classes))
        target_names = class_names
    # Classification report (per-class + macro & weighted)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=labels, target_names=target_names, digits=4, zero_division=0))

    # Micro & weighted averages
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
    prec_w, rec_w, f1_w, _      = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    print(f"Micro Avg  - Precision: {prec_micro:.4f}, Recall: {rec_micro:.4f}, F1: {f1_micro:.4f}")
    print(f"Weighted   - Precision: {prec_w:.4f}, Recall: {rec_w:.4f}, F1: {f1_w:.4f}")

    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(len(class_names), len(class_names)))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ticks = list(range(num_classes))
    ax.set_xticks(ticks)
    ax.set_xticklabels(class_names, rotation=0, ha='right')
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
    



