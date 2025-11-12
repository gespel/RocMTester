import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

# Device wählen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# 🧠 Einfaches Modell: 2 Eingaben (x, y) -> 1 Ausgabe (x + y)
model = nn.Sequential(
    nn.Linear(2, 8), 
    nn.ReLU(),
    nn.Linear(8, 8),
    nn.ReLU(),
    nn.Linear(8, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
).to(device)

opt = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# 📊 Trainingsdaten erzeugen
X = torch.rand(10000, 2, device=device) * 10  # Zufällige x, y Werte
y = X.sum(dim=1, keepdim=True)                # Ziel: x + y

# 🔁 Training
for epoch in tqdm.tqdm(range(100000)):
    opt.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    opt.step()


# 🧪 Test
test = torch.tensor([[2.0, 3.0], [5.5, 1.5], [10.0, 0.5]], device=device)
with torch.no_grad():
    result = model(test)
print("\nTest:")
for inp, out in zip(test, result):
    print(f"{inp.tolist()} -> {out.item():.3f}")