
import torch
import torch.nn as nn
import importlib.metadata as metadata

print("=" * 45)
print("SYNAPSENSE ENVIRONMENT TEST")
print("=" * 45)

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# Create sample input
X = torch.randn(32, 10)
y = torch.randint(0, 2, (32,))

# Build a simple neural network
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 2)
)

# Forward pass
outputs = model(X)

# Calculate loss
loss = nn.CrossEntropyLoss()(outputs, y)

# Backpropagation
loss.backward()

print("Input shape:", X.shape)
print("Output shape:", outputs.shape)
print("Training loss:", round(loss.item(), 4))
print("Gradients calculated:", model[0].weight.grad is not None)

print("=" * 45)
print("ENVIRONMENT TEST COMPLETED")
print("=" * 45)
