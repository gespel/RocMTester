import torch
import torch.nn as nn
import torch.optim as optim

# ⚙️ Gerät wählen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# 🎶 Beispiel: Wir wollen 4800-Sample lange 1D-Audios generieren
SAMPLE_LEN = 4800
LATENT_DIM = 64

# 🧩 Generator-Netzwerk (Decoder)
class VoiceGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, 512), nn.ReLU(),
            nn.Linear(512, 2048), nn.ReLU(),
            nn.Linear(2048, SAMPLE_LEN),
            nn.Tanh()  # Wertebereich [-1, 1]
        )

    def forward(self, z):
        return self.net(z)

# 🔧 Initialisierung
gen = VoiceGenerator().to(device)
opt = optim.Adam(gen.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# 📦 Dummy-Trainingsdaten (z. B. echte Sprachsamples, hier nur Zufallsrauschen)
real_audio = torch.randn(256, SAMPLE_LEN, device=device)

# 🎲 Training
for epoch in range(1000):
    # Zufälliger latenter Vektor
    z = torch.randn(256, LATENT_DIM, device=device)

    # Generiertes Audio
    fake_audio = gen(z)

    # Lernziel: das echte Signal rekonstruieren (Autoencoder-Idee)
    loss = loss_fn(fake_audio, real_audio)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoche {epoch+1}: Loss = {loss.item():.6f}")

# 🔉 Generiere 1 neue "Stimme"
z = torch.randn(1, LATENT_DIM, device=device)
voice = gen(z).squeeze().detach().cpu()

# Optional: als WAV speichern
import soundfile as sf
sf.write("gen_voice.wav", voice.numpy(), 48000)
print("✅ Gespeichert als gen_voice.wav")