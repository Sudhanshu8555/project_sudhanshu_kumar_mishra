import torch
import torch.optim as optim
import torch.nn as nn
from config import learning_rate, weight_decay, num_epochs, patience
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

def train(model, train_loader, val_loader):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_acc = 0.0
    counter = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        total_loss, correct = 0, 0
        for images, labels in tqdm(train_loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (output.argmax(1) == labels).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc = correct / len(train_loader.dataset)

        val_loss, val_acc = evaluate(model, val_loader)

        scheduler.step()
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/final_weights.pth")
            counter = 0
            print("🔥 New best model saved!")
        else:
            counter += 1
            print(f"😴 No improvement for {counter} epoch(s)")

        if counter >= patience:
            print("\n⏹️ Early stopping triggered")
            break

def evaluate(model, loader):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            total_loss += loss.item()
            correct += (output.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)
