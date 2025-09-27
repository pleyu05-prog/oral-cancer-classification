# scripts/train_imagefolder.py
import argparse, os, torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def get_loaders(data_dir, bs=32):
    mean, std = [0.485,0.456,0.406],[0.229,0.224,0.225]
    tf_train = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    tf_eval  = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), tf_train)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"),   tf_eval)
    test_ds  = datasets.ImageFolder(os.path.join(data_dir, "test"),  tf_eval)
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=2),
        DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=2),
        DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=2),
        train_ds.classes,
    )

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); correct=0; total=0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred==y).sum().item()
        total   += y.numel()
    return correct/total if total else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)  # e.g., data/oral-cancer-dataset
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="runs")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except Exception:
        model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)  # 二分类
    model = model.to(device)

    train_loader, val_loader, test_loader, classes = get_loaders(args.data_dir, args.batch_size)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    ce  = nn.CrossEntropyLoss()

    best=0.0
    for ep in range(1, args.epochs+1):
        model.train()
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            loss = ce(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {ep}: val_acc={val_acc:.4f}")
        if val_acc>best:
            best=val_acc
            torch.save({"state_dict": model.state_dict(), "val_acc": best},
                       os.path.join(args.out, "best.pt"))
            print(f">> new best {best:.4f}")
    test_acc = evaluate(model, test_loader, device)
    print(f"Test acc: {test_acc:.4f}")

if __name__ == "__main__":
    main()
