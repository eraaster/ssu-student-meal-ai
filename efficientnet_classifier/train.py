import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import timm
from torch.amp import autocast, GradScaler  # ✅ 최신 권장 방식

def main():
    # 1. 경로 설정
    data_dir = "C:\\Users\\soong\\Desktop\\한식이미지_sample"

    # 2. 고급 전처리 + 데이터 증강
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3)
    ])

    # 3. 데이터 로딩 및 분할
    full_dataset = ImageFolder(data_dir, transform=transform)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4, pin_memory=True)

    # 4. 모델 정의 (전이학습 + fine-tuning 확대)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("✅ Using device:", device)

    model = timm.create_model('efficientnet_b0', pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model = model.to(device)

    # 5. 더 넓은 범위 fine-tuning
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if any(x in name for x in ["blocks.2", "blocks.3", "blocks.4", "blocks.5", "blocks.6", "classifier"]):
            param.requires_grad = True

    # 6. 학습 설정 (AdamW + Cosine LR + Label Smoothing)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Mixed Precision
    scaler = GradScaler("cuda")

    # 7. 학습 루프
    best_acc = 0
    patience = 3
    trigger = 0  
    epochs = 20

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_batch = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast("cuda"):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_batch += 1

        avg_loss = total_loss / total_batch

        # 검증
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with autocast("cuda"):
                    outputs = model(imgs)
                pred = torch.argmax(outputs, dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        print(f"📊 Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Val Accuracy: {acc:.4f}")
        scheduler.step()

        if acc > best_acc:
            best_acc = acc
            trigger = 0
            torch.save(model.state_dict(), "main6.pt")
            print("✅ Best 모델 저장 완료!")
        else:
            trigger += 1
            if trigger >= patience:
                print("🛑 EarlyStopping 발생")
                break

# 🔐 Windows multiprocessing 안전 실행용
if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()
    main()
