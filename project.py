try:
    import torch
    import torchvision
    assert int(torch.__version__.split(".")[1]) >= 12
    assert int(torchvision.__version__.split(".")[1]) >= 13
    print(f"torch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")
except:
    print("[INFO] torch/torchvision versions not as required, installing nightly versions.")
    import subprocess
    subprocess.run(["pip3", "install", "-U", "torch", "torchvision", "torchaudio",
                    "--extra-index-url", "https://download.pytorch.org/whl/cu113"])
    import torch
    import torchvision
    print(f"torch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


try:
    from going_modular.going_modular import data_setup, engine
except:
    print("[INFO] Couldn't find going_modular scripts... downloading them from GitHub.")
    import subprocess
    subprocess.run(["git", "clone", "https://github.com/mrdbourke/pytorch-deep-learning"])
    subprocess.run(["mv", "pytorch-deep-learning/going_modular", "."])
    subprocess.run(["rm", "-rf", "pytorch-deep-learning"])
    from going_modular.going_modular import data_setup, engine


import os
import subprocess
from pathlib import Path

def find_dataset_root(base: Path) -> Path:
    
    import shutil
    macosx = base / "__MACOSX"
    if macosx.exists():
        shutil.rmtree(macosx)
        print("[INFO] __MACOSX deleted!")
    for junk in base.rglob("._*"):
        junk.unlink(missing_ok=True)

    candidates = []
    for p in sorted(base.rglob("train")):
        if not p.is_dir() or "__MACOSX" in str(p):
            continue
        images = list(p.rglob("*.jpeg")) + list(p.rglob("*.jpg")) + list(p.rglob("*.png"))
        if images:
            candidates.append((len(images), p))

    if not candidates:
        raise FileNotFoundError(f"The folder containing images, 'train', could not be found.: {base}")

    candidates.sort(reverse=True)
    best = candidates[0][1]
    print(f"[INFO] Dataset folder: {best.parent} ({candidates[0][0]} image")
    return best.parent

def upload_kaggle_json():
    from google.colab import files
    import shutil
    print("[INFO] choose the kaggle.json...")
    files.upload()
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    shutil.copy("kaggle.json", os.path.expanduser("~/.kaggle/kaggle.json"))
    os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)
    print("[INFO] kaggle.json configured!")

def download_dataset_kaggle() -> Path:
    upload_kaggle_json()
    subprocess.run(["pip", "install", "-q", "kaggle"], check=True)
    zip_path = Path("chest-xray-pneumonia.zip")
    if not zip_path.exists():
        subprocess.run(["kaggle", "datasets", "download", "-d",
                        "paultimothymooney/chest-xray-pneumonia"], check=True)
        print("[INFO] Dataset downloaded!")
    else:
        print("[INFO] Zip is already downloaded, passing the downloading.")
    out_dir = Path("chest_xray_dataset")
    if not out_dir.exists():
        result = subprocess.run(["unzip", "-q", str(zip_path), "-d", str(out_dir)])
        if result.returncode not in (0, 1):
            raise RuntimeError(f"unzip failed: {result.returncode}")
        print("[INFO] Zip opened!")
    else:
        print("[INFO] chest_xray_dataset is already downloaded, passing the downloading.")
    return find_dataset_root(out_dir)


extract_dir_path = download_dataset_kaggle()


train_dir = extract_dir_path / "train"
test_dir  = extract_dir_path / "test"
print(f"Train dir: {train_dir} | Test dir: {test_dir}")


import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from PIL import Image, UnidentifiedImageError
from going_modular.going_modular.engine import train_step, test_step


vit_weights     = torchvision.models.ViT_B_16_Weights.DEFAULT
vit_transform   = vit_weights.transforms()


effnet_weights    = torchvision.models.EfficientNet_B0_Weights.DEFAULT
effnet_transform  = effnet_weights.transforms()

train_augment_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except (UnidentifiedImageError, OSError):
            return self.__getitem__((index + 1) % len(self))

def get_dataloaders(train_transform, test_transform, batch_size=32):
    train_data = SafeImageFolder(root=train_dir, transform=train_transform)
    test_data  = SafeImageFolder(root=test_dir,  transform=test_transform)
    class_names = train_data.classes
    print(f"Classes: {class_names} | Train: {len(train_data)} | Test: {len(test_data)}")
    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=True)
    test_dl  = DataLoader(test_data,  batch_size=batch_size, shuffle=False,
                          num_workers=2, pin_memory=True)
    return train_dl, test_dl, class_names

def create_writer(experiment_name: str, model_name: str, extra: str = None) -> SummaryWriter:
    timestamp = datetime.now().strftime("%Y-%m-%d")
    log_dir = os.path.join("runs", timestamp, experiment_name, model_name,
                           extra if extra else "")
    print(f"[INFO] TensorBoard log: {log_dir}")
    return SummaryWriter(log_dir=log_dir)


def build_vit(num_classes: int, dropout: float = 0.3) -> nn.Module:
    weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    model = torchvision.models.vit_b_16(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    torch.manual_seed(42)
    model.heads = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(768, num_classes)
    )
    return model

def build_efficientnet(num_classes: int, dropout: float = 0.3) -> nn.Module:
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    torch.manual_seed(42)
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(1280, num_classes)
    )
    return model

class EarlyStopping:
    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_acc   = 0.0
        self.should_stop = False

    def step(self, val_acc: float) -> bool:
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc = val_acc
            self.counter  = 0
        else:
            self.counter += 1
            print(f"[EarlyStopping] No improvement for {self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train(model: nn.Module,
          train_dataloader: DataLoader,
          test_dataloader: DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: nn.Module,
          epochs: int,
          device: torch.device,
          writer: SummaryWriter = None,
          early_stopping: Optional[EarlyStopping] = None) -> Dict[str, List]:

    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    model.to(device)

    best_acc        = 0.0
    best_state_dict = None

    if writer:
        try:
            writer.add_graph(model=model,
                             input_to_model=torch.rand(32, 3, 224, 224).to(device))
        except Exception as e:
            print(f"[WARN] add_graph failed: {e}")

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model, dataloader=train_dataloader,
                                           loss_fn=loss_fn, optimizer=optimizer, device=device)
        test_loss, test_acc   = test_step(model=model, dataloader=test_dataloader,
                                           loss_fn=loss_fn, device=device)

        print(f"Epoch: {epoch+1} | "
              f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
              f"test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}")

      
        if test_acc > best_acc:
            best_acc        = test_acc
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  --> New best test acc: {best_acc:.4f} (model is saved.)")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        if writer:
            writer.add_scalars("Loss",     {"train_loss": train_loss, "test_loss": test_loss}, epoch)
            writer.add_scalars("Accuracy", {"train_acc":  train_acc,  "test_acc":  test_acc},  epoch)

    
        if early_stopping and early_stopping.step(test_acc):
            print(f"[INFO] Early stopping! Best acc: {best_acc:.4f}")
            break

    if writer:
        writer.close()
        print("[INFO] TensorBoard writer closed.")


    if best_state_dict:
        model.load_state_dict(best_state_dict)

    results["best_test_acc"] = best_acc
    return results


print("\n" + "="*50)
print("EXPERIMENT 1: ViT-B/16")
print("="*50)

train_dl_vit, test_dl_vit, class_names = get_dataloaders(
    train_transform=vit_transform,
    test_transform=vit_transform
)

vit_model     = build_vit(num_classes=len(class_names), dropout=0.3)
vit_optimizer = torch.optim.Adam(vit_model.parameters(), lr=0.001)
vit_loss_fn   = nn.CrossEntropyLoss()
vit_writer    = create_writer("chest_xray", "vit_b16", "10_epochs_earlystop")
vit_es        = EarlyStopping(patience=3)

vit_results = train(
    model=vit_model,
    train_dataloader=train_dl_vit,
    test_dataloader=test_dl_vit,
    optimizer=vit_optimizer,
    loss_fn=vit_loss_fn,
    epochs=10,
    device=device,
    writer=vit_writer,
    early_stopping=vit_es
)

print("\n" + "="*50)
print("EXPERIMENT 2: EfficientNet-B0")
print("="*50)

train_dl_eff, test_dl_eff, _ = get_dataloaders(
    train_transform=train_augment_transform,
    test_transform=effnet_transform
)

eff_model     = build_efficientnet(num_classes=len(class_names), dropout=0.3)
eff_optimizer = torch.optim.Adam(eff_model.parameters(), lr=0.001)
eff_loss_fn   = nn.CrossEntropyLoss()
eff_writer    = create_writer("chest_xray", "efficientnet_b0", "10_epochs_earlystop")
eff_es        = EarlyStopping(patience=3)

eff_results = train(
    model=eff_model,
    train_dataloader=train_dl_eff,
    test_dataloader=test_dl_eff,
    optimizer=eff_optimizer,
    loss_fn=eff_loss_fn,
    epochs=10,
    device=device,
    writer=eff_writer,
    early_stopping=eff_es
)

print("\n" + "="*50)
print("RESULTS")
print("="*50)
print(f"ViT-B/16       - Best Test Acc: {vit_results['best_test_acc']:.4f}")
print(f"EfficientNet-B0 - Best Test Acc: {eff_results['best_test_acc']:.4f}")

winner = "ViT-B/16" if vit_results['best_test_acc'] >= eff_results['best_test_acc'] else "EfficientNet-B0"
print(f"\nBest model: {winner}")

save_path = Path("models")
save_path.mkdir(exist_ok=True)
torch.save(vit_model.state_dict(),     save_path / "vit_b16_best.pth")
torch.save(eff_model.state_dict(),     save_path / "efficientnet_b0_best.pth")
print(f"[INFO] Models are saved: {save_path}/")

%load_ext tensorboard
%tensorboard --logdir runs

