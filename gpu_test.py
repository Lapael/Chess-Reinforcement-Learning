import torch

print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))

# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
