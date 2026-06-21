"""
Проверяем, активируются ли switch-нейроны на тестовых промптах
"""
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
TEST_PROMPTS = [
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "Calculate: 24578 + 13892 =",
    "The capital of France is",
]

# Загружаем модель
print("Loading model...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16)
model.eval()

# Загружаем switches
with open("checkpoints/switches.json") as f:
    data = json.load(f)

# Берём top-100 нейронов по score для каждого слоя
switches = {}
for k, v in data["switches"].items():
    sorted_switches = sorted(v, key=lambda x: x[1], reverse=True)
    switches[int(k)] = [n for n, s in sorted_switches[:100]]

# Собираем активации
print("\nCollecting activations...")
activation_counts = {layer: {n: 0 for n in neurons} for layer, neurons in switches.items()}

for prompt in TEST_PROMPTS:
    inputs = tok(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        
        # Проверяем активации MLP для каждого слоя
        for layer_idx in range(28):
            hidden_states = outputs.hidden_states[layer_idx + 1]  # +1 потому что hidden_states[0] = embeddings
            
            # Получаем gate_proj активации
            layer = model.model.layers[layer_idx]
            gate_out = layer.mlp.gate_proj(hidden_states)
            
            # Проверяем, какие нейроны активны (ReLU > 0)
            active_neurons = (gate_out > 0).any(dim=1).squeeze().cpu().numpy()
            
            for neuron_idx in switches[layer_idx]:
                if neuron_idx < len(active_neurons) and active_neurons[neuron_idx]:
                    activation_counts[layer_idx][neuron_idx] += 1

# Анализируем результаты
print("\n" + "=" * 70)
print("АКТИВАЦИЯ SWITCH-НЕЙРОНОВ НА ТЕСТОВЫХ ПРОМПТАХ")
print("=" * 70)

total_switches = sum(len(neurons) for neurons in switches.values())
activated_switches = sum(1 for layer in activation_counts.values() 
                         for count in layer.values() if count > 0)

print(f"\nВсего switch-нейронов: {total_switches}")
print(f"Активировались хотя бы раз: {activated_switches} ({100*activated_switches/total_switches:.1f}%)")

# По слоям
print(f"\n{'Layer':<8} {'Switches':<12} {'Activated':<12} {'Percentage':<12}")
print("-" * 70)
for layer_idx in range(28):
    total = len(switches[layer_idx])
    activated = sum(1 for count in activation_counts[layer_idx].values() if count > 0)
    pct = 100 * activated / total if total > 0 else 0
    print(f"{layer_idx:<8} {total:<12} {activated:<12} {pct:.1f}%")

# Top-10 самых активируемых нейронов
print(f"\n{'=' * 70}")
print("TOP-10 САМЫХ АКТИВИРУЕМЫХ SWITCH-НЕЙРОНОВ")
print("=" * 70)

all_activations = []
for layer_idx, neurons in activation_counts.items():
    for neuron_idx, count in neurons.items():
        if count > 0:
            all_activations.append((layer_idx, neuron_idx, count))

all_activations.sort(key=lambda x: x[2], reverse=True)
for layer_idx, neuron_idx, count in all_activations[:10]:
    print(f"Layer {layer_idx:2d}, Neuron {neuron_idx:4d}: активировался {count} раз")

print(f"\n{'=' * 70}")
print("ВЫВОД:")
print("=" * 70)
if activated_switches / total_switches < 0.5:
    print("⚠️  Менее 50% switch-нейронов активируются на тестовых промптах!")
    print("Это значит, что мы восстанавливаем веса для нейронов, которые не используются.")
    print("Возможно, switch finder находит 'потенциально важные' нейроны,")
    print("но они не активируются на данных промптах.")
else:
    print("✓ Большинство switch-нейронов активируются")
