import torch
import torchvision

print(f"PyTorch version: {torch.__version__}")
print(f"torchvision version: {torchvision.__version__}")

# Test the problematic import
import torchvision.transforms.functional as tvisf
print("torchvision.transforms.functional imported successfully")

# Test nms function if available
try:
    from torchvision.ops import nms
    print("torchvision.ops.nms available")
except Exception as e:
    print(f"Error with nms: {e}")
