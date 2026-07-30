import torch
import subprocess
import json
import requests

print("=" * 60)
print("GPU STATUS CHECK - DeepSeek-R1:7B")
print("=" * 60)

# 1. Check PyTorch CUDA
print("\n1. PyTorch CUDA Status:")
print(f"   CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU Count: {torch.cuda.device_count()}")
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"   GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"   GPU Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

# 2. Check Ollama
print("\n2. Ollama Status:")
try:
    response = requests.get("http://localhost:11434/api/ps", timeout=5)
    if response.status_code == 200:
        data = response.json()
        print(f"   Running Models: {len(data.get('models', []))}")
        for model in data.get('models', []):
            print(f"   - {model.get('name')}")
            if model.get('details', {}).get('gpu'):
                print(f"     ✓ GPU: {model['details']['gpu']}")
    else:
        print(f"   Ollama API returned: {response.status_code}")
except Exception as e:
    print(f"   ❌ Ollama not running: {e}")

# 3. Check NVIDIA