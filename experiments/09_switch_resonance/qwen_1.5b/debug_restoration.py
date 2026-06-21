"""
Debug: проверка правильности weight restoration
"""
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B"

# Загружаем модель
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16)
model.eval()

# Берём один нейрон для проверки
layer_idx = 0
neuron_idx = 1215  # Top-1 нейрон
mod_name = "mlp.gate_proj"

# Получаем оригинальные веса
layer = model.model.layers[layer_idx]
mod = layer
for part in mod_name.split('.'):
    mod = getattr(mod, part)
original_weight = mod.weight[:, neuron_idx].clone()

print(f"\nОригинальные веса для Layer {layer_idx}, Neuron {neuron_idx}, {mod_name}:")
print(f"  Shape: {original_weight.shape}")
print(f"  Mean: {original_weight.mean():.6f}")
print(f"  Std: {original_weight.std():.6f}")
print(f"  Min: {original_weight.min():.6f}")
print(f"  Max: {original_weight.max():.6f}")
print(f"  First 5 values: {original_weight[:5].tolist()}")

# Добавляем шум
print(f"\nДобавляем шум...")
noise = torch.randn_like(original_weight) * 0.02
noisy_weight = original_weight + noise

print(f"\nЗашумлённые веса:")
print(f"  Mean: {noisy_weight.mean():.6f}")
print(f"  Std: {noisy_weight.std():.6f}")
print(f"  Min: {noisy_weight.min():.6f}")
print(f"  Max: {noisy_weight.max():.6f}")
print(f"  First 5 values: {noisy_weight[:5].tolist()}")

# Проверяем разницу
diff = (noisy_weight - original_weight).abs()
print(f"\nРазница (noisy - original):")
print(f"  Mean diff: {diff.mean():.6f}")
print(f"  Max diff: {diff.max():.6f}")

# Восстанавливаем
print(f"\nВосстанавливаем веса...")
mod.weight.data[:, neuron_idx] = original_weight
restored_weight = mod.weight[:, neuron_idx].clone()

print(f"\nВосстановленные веса:")
print(f"  Mean: {restored_weight.mean():.6f}")
print(f"  Std: {restored_weight.std():.6f}")
print(f"  Min: {restored_weight.min():.6f}")
print(f"  Max: {restored_weight.max():.6f}")
print(f"  First 5 values: {restored_weight[:5].tolist()}")

# Проверяем, что восстановление правильное
diff_restored = (restored_weight - original_weight).abs()
print(f"\nРазница (restored - original):")
print(f"  Mean diff: {diff_restored.mean():.6f}")
print(f"  Max diff: {diff_restored.max():.6f}")

if diff_restored.max() < 1e-5:
    print("\n✓ Восстановление работает ПРАВИЛЬНО")
else:
    print("\n✗ Восстановление работает НЕПРАВИЛЬНО!")
