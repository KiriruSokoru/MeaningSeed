"""
Minimal Restoration — восстанавливаем только top-10 нейронов на слой
"""
import json, torch, gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
TEST_PROMPTS = [
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "Calculate: 24578 + 13892 =",
    "The capital of France is",
]

INPUT_MODULES = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
                 'mlp.gate_proj', 'mlp.up_proj']
OUTPUT_MODULES = ['self_attn.o_proj', 'mlp.down_proj']

def add_noise(model, noise_std):
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                noise = torch.randn_like(p) * noise_std
                p.add_(noise)

def save_minimal(model, switches):
    """Сохраняет веса только для top-K нейронов"""
    saved = {}
    with torch.no_grad():
        for layer_idx, switch_list in switches.items():
            layer = model.model.layers[int(layer_idx)]
            for neuron_idx, score in switch_list:
                for mod_name in INPUT_MODULES:
                    mod = layer
                    for part in mod_name.split('.'):
                        mod = getattr(mod, part)
                    w = mod.weight
                    key = f"model.layers.{layer_idx}.{mod_name}.weight"
                    if w.dim() >= 2 and w.shape[1] > neuron_idx:
                        if key not in saved:
                            saved[key] = {}
                        saved[key][neuron_idx] = w[:, neuron_idx].clone().cpu()
                
                for mod_name in OUTPUT_MODULES:
                    mod = layer
                    for part in mod_name.split('.'):
                        mod = getattr(mod, part)
                    w = mod.weight
                    key = f"model.layers.{layer_idx}.{mod_name}.weight"
                    if w.dim() >= 2 and w.shape[0] > neuron_idx:
                        if key not in saved:
                            saved[key] = {}
                        saved[key][neuron_idx] = w[neuron_idx, :].clone().cpu()
    return saved

def restore_minimal(model, saved, switches):
    """Восстанавливает только top-K нейронов"""
    param_count = 0
    with torch.no_grad():
        for layer_idx, switch_list in switches.items():
            layer = model.model.layers[int(layer_idx)]
            for neuron_idx, score in switch_list:
                for mod_name in INPUT_MODULES:
                    mod = layer
                    for part in mod_name.split('.'):
                        mod = getattr(mod, part)
                    w = mod.weight
                    key = f"model.layers.{layer_idx}.{mod_name}.weight"
                    if key in saved and neuron_idx in saved[key]:
                        w.data[:, neuron_idx] = saved[key][neuron_idx].to(w.device)
                        param_count += w.shape[0]
                
                for mod_name in OUTPUT_MODULES:
                    mod = layer
                    for part in mod_name.split('.'):
                        mod = getattr(mod, part)
                    w = mod.weight
                    key = f"model.layers.{layer_idx}.{mod_name}.weight"
                    if key in saved and neuron_idx in saved[key]:
                        w.data[neuron_idx, :] = saved[key][neuron_idx].to(w.device)
                        param_count += w.shape[1]
    return param_count

def compute_ppl(model, tok, texts):
    model.eval()
    total_loss, total_tokens = 0, 0
    with torch.no_grad():
        for text in texts:
            inputs = tok(text, return_tensors="pt", max_length=128, truncation=True)
            out = model(**inputs, labels=inputs["input_ids"])
            total_loss += out.loss.item() * inputs["input_ids"].numel()
            total_tokens += inputs["input_ids"].numel()
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()

def main():
    console.print(Panel.fit(
        f"[bold cyan]Minimal Restoration — {MODEL_NAME}[/bold cyan]\n"
        f"[yellow]Тестируем top-5, top-10, top-20 нейронов на слой[/yellow]",
        title="🎯 Minimal Restoration", box=box.DOUBLE_EDGE
    ))

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Загружаем switches и сортируем по score
    with open("checkpoints/switches.json") as f:
        data = json.load(f)
    
    all_switches = {}
    for k, v in data["switches"].items():
        sorted_switches = sorted(v, key=lambda x: x[1], reverse=True)
        all_switches[int(k)] = sorted_switches
    
    # Тестируем разные top-K
    noise_std = 0.02
    results = []
    
    for top_k in [5, 10, 20, 50]:
        console.print(f"\n[cyan]Тестируем top-{top_k} нейронов...[/cyan]")
        
        # Берём top-K нейронов
        switches = {k: v[:top_k] for k, v in all_switches.items()}
        
        # Загружаем модель
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16)
        model.eval()
        
        # Сохраняем веса
        saved = save_minimal(model, switches)
        
        # Baseline
        base_ppl = compute_ppl(model, tok, TEST_PROMPTS)
        
        # Noise
        add_noise(model, noise_std)
        noisy_ppl = compute_ppl(model, tok, TEST_PROMPTS)
        
        # Restore
        count = restore_minimal(model, saved, switches)
        rest_ppl = compute_ppl(model, tok, TEST_PROMPTS)
        
        improvement = noisy_ppl / rest_ppl if rest_ppl > 0 else float('inf')
        
        results.append({
            "top_k": top_k,
            "restored_params": count,
            "baseline_ppl": base_ppl,
            "noisy_ppl": noisy_ppl,
            "restored_ppl": rest_ppl,
            "improvement": improvement
        })
        
        console.print(f"  Top-{top_k}: {count:10,} params → PPL {rest_ppl:12.2f} ({improvement:8.1f}x)")
        
        del model, saved
        gc.collect()
    
    # Таблица
    table = Table(title="📊 Minimal Restoration Results", box=box.ROUNDED)
    table.add_column("Top-K", style="cyan", justify="right")
    table.add_column("Params", style="magenta", justify="right")
    table.add_column("Noisy PPL", style="red", justify="right")
    table.add_column("Restored PPL", style="green", justify="right")
    table.add_column("Improvement", style="yellow", justify="right")
    
    for r in results:
        table.add_row(
            f"{r['top_k']}",
            f"{r['restored_params']:,}",
            f"{r['noisy_ppl']:.2f}",
            f"{r['restored_ppl']:.2f}",
            f"{r['improvement']:.1f}x"
        )
    
    console.print("\n")
    console.print(table)

if __name__ == "__main__":
    main()
