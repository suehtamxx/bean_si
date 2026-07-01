import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets


DATA_DIR    = r"dataverse_files\Classification\Classification"
IMG_SIZE    = 224
BATCH_SIZE         = 32
EPOCHS             = 100
EARLY_STOP_PATIENCE = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {DEVICE}")


train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


train_ds = datasets.ImageFolder(root=f"{DATA_DIR}/training",   transform=train_transform)
val_ds   = datasets.ImageFolder(root=f"{DATA_DIR}/validation", transform=val_transform)
test_ds  = datasets.ImageFolder(root=f"{DATA_DIR}/test",       transform=val_transform)

NUM_CLASSES = len(train_ds.classes)

print(f"\nClasses encontradas ({NUM_CLASSES}): {train_ds.classes}")
print(f"Treino:    {len(train_ds):>5} imagens")
print(f"Validação: {len(val_ds):>5} imagens")
print(f"Teste:     {len(test_ds):>5} imagens")

_pin     = DEVICE.type == "cuda"
_workers = 0 if sys.platform == "win32" else 2
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=_workers, pin_memory=_pin)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=_workers, pin_memory=_pin)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=_workers, pin_memory=_pin)


class BeanCNN(nn.Module):
    def __init__(self, num_classes):
        super(BeanCNN, self).__init__()

        self.features = nn.Sequential(
            # Bloco 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 224 → 112

            # Bloco 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 112 → 56

            # Bloco 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 56 → 28
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = BeanCNN(num_classes=NUM_CLASSES).to(DEVICE)
print(f"\n{model}\n")


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


def train_epoch(model, loader):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted  = outputs.max(1)
        correct       += predicted.eq(labels).sum().item()
        total         += labels.size(0)

    return running_loss / total, correct / total


def eval_epoch(model, loader):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted  = outputs.max(1)
            correct       += predicted.eq(labels).sum().item()
            total         += labels.size(0)

    return running_loss / total, correct / total


print("Iniciando o treinamento...")
print(f"{'Época':>6} | {'Loss Treino':>11} | {'Acc Treino':>10} | {'Loss Val':>8} | {'Acc Val':>7}")
print("-" * 60)

best_val_acc     = 0.0
early_stop_count = 0

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_epoch(model, train_loader)
    val_loss,   val_acc   = eval_epoch(model, val_loader)
    scheduler.step()

    print(f"{epoch:>6} | {train_loss:>11.4f} | {train_acc*100:>9.2f}% | "
          f"{val_loss:>8.4f} | {val_acc*100:>6.2f}%")

    # Salva o melhor modelo
    if val_acc > best_val_acc:
        best_val_acc     = val_acc
        early_stop_count = 0
        torch.save(model.state_dict(), "bean_disease_cnn_model_best.pth")
        print(f"         → Novo melhor modelo salvo! (val_acc={best_val_acc*100:.2f}%)")
    else:
        early_stop_count += 1
        print(f"         → Sem melhora ({early_stop_count}/{EARLY_STOP_PATIENCE})")
        if early_stop_count >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping ativado na época {epoch}. Melhor val_acc: {best_val_acc*100:.2f}%")
            break


model.load_state_dict(torch.load("bean_disease_cnn_model_best.pth", map_location=DEVICE, weights_only=True))
test_loss, test_acc = eval_epoch(model, test_loader)
print(f"\nResultado no conjunto de TESTE → Loss: {test_loss:.4f} | Acurácia: {test_acc*100:.2f}%")

torch.save(model.state_dict(), "bean_disease_cnn_model.pth")
print("Modelo final salvo em: bean_disease_cnn_model.pth")