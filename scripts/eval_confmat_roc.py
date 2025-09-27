# scripts/eval_confmat_roc.py
import os, argparse, torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import numpy as np

def get_loader(split_dir, bs=64):
    mean, std = [0.485,0.456,0.406],[0.229,0.224,0.225]
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    ds = datasets.ImageFolder(split_dir, transform=tf)
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2), ds.classes

@torch.no_grad()
def collect(model, loader, device):
    ys, yps, yprobs = [], [], []
    model.eval()
    for x, y in loader:
        x = x.to(device); y = y.numpy()
        logits = model(x).cpu()
        prob = torch.softmax(logits, dim=1).numpy()
        yp = prob.argmax(1)
        ys.append(y); yps.append(yp); yprobs.append(prob)
    return np.concatenate(ys), np.concatenate(yps), np.concatenate(yprobs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/oral_cancer_split")
    ap.add_argument("--ckpt", type=str, default="runs/best.pt")
    ap.add_argument("--out_dir", type=str, default="assets")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 用 test 子集评估
    test_dir = Path(args.data_dir) / "test"
    loader, classes = get_loader(test_dir, bs=64)
    n_classes = len(classes)

    # 模型结构与训练时一致
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, n_classes)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)  # 兼容保存键
    model.load_state_dict(state, strict=False)
    model = model.to(device)

    y_true, y_pred, y_prob = collect(model, loader, device)

    # 1) 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig_cm, ax_cm = plt.subplots(figsize=(4,4))
    disp.plot(ax=ax_cm, cmap="Blues", colorbar=False, values_format="d")
    fig_cm.tight_layout()
    cm_path = os.path.join(args.out_dir, "confusion_matrix.png")
    fig_cm.savefig(cm_path, dpi=150)
    plt.close(fig_cm)

    # 2) ROC（仅二分类）
    if n_classes == 2:
        # 以类别名包含 'CANCER' 的为阳性；若找不到就取索引1
        pos_idx = 0 if any("CANCER" in c.upper() for c in [classes[0]]) else 1
        fpr, tpr, _ = roc_curve(y_true == pos_idx, y_prob[:, pos_idx])
        roc_auc = auc(fpr, tpr)
        fig_roc, ax_roc = plt.subplots(figsize=(4,4))
        ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax_roc.plot([0,1],[0,1],"--")
        ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR"); ax_roc.set_title("ROC Curve")
        ax_roc.legend(loc="lower right")
        fig_roc.tight_layout()
        roc_path = os.path.join(args.out_dir, "roc.png")
        fig_roc.savefig(roc_path, dpi=150)
        plt.close(fig_roc)
        print(f"Saved: {cm_path}, {roc_path}")
    else:
        print(f"Saved: {cm_path} (ROC skipped for multi-class)")

if __name__ == "__main__":
    main()
