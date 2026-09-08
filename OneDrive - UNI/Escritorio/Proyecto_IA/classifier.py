"""
Esqueleto de clasificador secundario (PyTorch) para verificación de crops.
Usa estructura de carpetas `dataset/phone` y `dataset/not_phone` (ImageFolder compatible).
"""
import argparse
import os
from pathlib import Path

import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def get_dataloaders(data_dir, batch_size=16, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    train_ds = datasets.ImageFolder(os.path.join(data_dir), transform=transform)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    return loader


def build_model(num_classes=2):
    # usar ResNet18 como punto de partida
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_loop(data_dir, epochs=5, lr=1e-4, batch_size=16, img_size=224, out='classifier.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = get_dataloaders(data_dir, batch_size=batch_size, img_size=img_size)
    model = build_model(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train()
        total, correct = 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            outp = model(xb)
            loss = criterion(outp, yb)
            loss.backward()
            opt.step()
            preds = outp.argmax(dim=1)
            total += yb.size(0)
            correct += (preds == yb).sum().item()
        print(f'Ep {ep+1}/{epochs} acc={(correct/total):.3f}')

    torch.save(model.state_dict(), out)
    print('Modelo guardado en', out)


class Verifier:
    """Carga un modelo entrenado (ResNet18 fc) para inferencia rápida sobre crops.
    Espera que las clases de ImageFolder sean ['not_phone','phone'] por orden alfabético.
    """
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = build_model(num_classes=2)
        try:
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)
        except Exception as e:
            raise RuntimeError(f'No se pudo cargar verificador: {e}')
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def predict_prob(self, crop_np) -> float:
        """Devuelve probabilidad de clase 'phone' (0..1)."""
        with torch.no_grad():
            try:
                x = self.transform(crop_np)
            except Exception:
                # fallback: convert via numpy->tensor
                import numpy as np
                arr = np.asarray(crop_np)
                arr = cv2.resize(arr, (224, 224))
                x = transforms.ToTensor()(arr)
            x = x.unsqueeze(0).to(self.device)
            out = self.model(x)
            probs = torch.softmax(out, dim=1)
            # index 1 is 'phone' (ImageFolder alphabetical: not_phone, phone)
            prob = float(probs[0, 1].cpu().item())
            return prob

    def predict_probs(self, crops: list, batch_size: int = 16) -> list:
        """Procesa una lista de crops (np arrays RGB/BGR) y devuelve lista de probabilidades.
        Usa batching y el device configurado para acelerar cuando haya GPU.
        """
        results = []
        n = len(crops)
        i = 0
        with torch.no_grad():
            while i < n:
                batch = crops[i:i+batch_size]
                tensors = []
                for crop in batch:
                    try:
                        t = self.transform(crop)
                    except Exception:
                        import numpy as np
                        arr = np.asarray(crop)
                        arr = cv2.resize(arr, (224, 224))
                        t = transforms.ToTensor()(arr)
                    tensors.append(t)
                x = torch.stack(tensors, dim=0).to(self.device)
                out = self.model(x)
                probs = torch.softmax(out, dim=1)[:, 1].cpu().tolist()
                results.extend([float(p) for p in probs])
                i += batch_size
        return results



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='dataset', help='Carpeta con subcarpetas phone/ not_phone')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--out', default='classifier.pth')
    args = parser.parse_args()
    train_loop(args.data, epochs=args.epochs, out=args.out)


if __name__ == '__main__':
    main()
