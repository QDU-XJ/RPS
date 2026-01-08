import torch
import torch.nn as nn
from model import MultiModalClassifier
model = MultiModalClassifier()
total = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total / 1e6:.2f} M")