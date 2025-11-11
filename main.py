import torch, torch.nn as nn, torch.optim as optim, torchvision as tv, time
device = torch.device("cuda"); print("Device:", device)

# ⚙️ Torch-Optimierungen
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

# Dataset – größere Batches + paralleles Laden
tfm = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5,), (0.5,))
])
train = tv.datasets.FashionMNIST("./data", train=True, download=True, transform=tfm)
test = tv.datasets.FashionMNIST("./data", train=False, download=True, transform=tfm)
train_loader = torch.utils.data.DataLoader(train, batch_size=2048, shuffle=True, num_workers=8, pin_memory=True)
test_loader  = torch.utils.data.DataLoader(test, batch_size=4096, num_workers=8, pin_memory=True)

# Modell – größer, um GPU auszulasten
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 2048), nn.GELU(),
    nn.Linear(2048, 1024), nn.GELU(),
    nn.Linear(1024, 512), nn.GELU(),
    nn.Linear(512, 10)
).to(device)

model = torch.compile(model)  # 🔥 nutzt TorchDynamo für GPU-Optimierung

opt = optim.AdamW(model.parameters(), lr=2e-3)
loss_fn = nn.CrossEntropyLoss()

# Training
for e in range(50):
    model.train(); t0 = time.time()
    for x, y in train_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    print(f"Epoche {e+1}: Loss {loss.item():.4f}, Zeit {time.time()-t0:.1f}s")

# Test
model.eval(); correct = total = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item(); total += y.size(0)
print(f"✅ Genauigkeit: {100*correct/total:.2f}%")
