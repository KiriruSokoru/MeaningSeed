"""
Selective Restoration — восстанавливаем только верхние слои (все модули)
"""
import json, torch, gc
from pathlib import Path
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
    "Write a Python function to calculate factorial:",
    "Summarize: Artificial intelligence is transforming industries.",
]

INPUT_MODULES = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
                 'mlp.gate_proj', 'mlp.up_proj']
OUTPUT_MODULES = ['self_attn.o_proj', 'mlp.down_proj']
ALL_MODULES = INPUT_MODULES + OUTPUT_MODULES

def add_noise(model, noise_std):
    """Добавляет шум ко всем параметрам"""
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                noise = torch.randn_like(p) * noise_std
                p.add_(noise)

def save_all_modules(model, switches):
    """Сохраняет веса ВСЕХ модулей для switches"""
    saved = {}
    with torch.no_grad():
        for layer_idx, switch_list in switches.items():
            layer = model.model.layers[int(layer_idx)]
            for neuron_idx, score in switch_list:
                # Input modules (weights[:, neuron_idx])
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
                
                # Output modules (weights[neuron_idx, :])
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
    gc.collect()
    return saved

def restore_selective(model, saved, switches, start_layer=0):
    """Восстанавливает все модули для слоёв >= start_layer"""
    param_count = 0
    with torch.no_grad():
        for layer_idx, switch_list in switches.items():
            if int(layer_idx) < start_layer:
                continue  # Пропускаем нижние слои
            
            layer = model.model.layers[int(layer_idx)]
            for neuron_idx, score in switch_list:
                # Restore input modules
                for mod_name in INPUT_MODULES:
                    mod = layer
                    for part in mod_name.split('.'):
                        mod = getattr(mod, part)
                    w = mod.weight
                    key = f"model.layers.{layer_idx}.{mod_name}.weight"
                    if key in saved and neuron_idx in saved[key]:
                        w.data[:, neuron_idx] = saved[key][neuron_idx].to(w.device)
                        param_count += w.shape[0]  # Размер столбца
                
                # Restore output modules
                for mod_name in OUTPUT_MODULES:
                    mod = layer
                    for part in mod_name.split('.'):
                        mod = getattr(mod, part)
                    w = mod.weight
                    key = f"model.layers.{layer_idx}.{mod_name}.weight"
                    if key in saved and neuron_idx in saved[key]:
                        w.data[neuron_idx, :] = saved[key][neuron_idx].to(w.device)
                        param_count += w.shape[1]  # Размер строки
    gc.collect()
    return param_count

def compute_ppl(model, tok, texts, max_len=128):
    """Вычисляет PPL"""
    model.eval()
    total_loss, total_tokens = 0, 0
    with torch.no_grad():
        for text in texts:
            inputs = tok(text, return_tensors="pt", max_length=max_len, truncation=True)
            out = model(**inputs, labels=inputs["input_ids"])
            total_loss += out.loss.item() * inputs["input_ids"].numel()
            total_tokens += inputs["input_ids"].numel()
            del inputs, out
            gc.collect()
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()

def main():
    console.print(Panel.fit(
        f"[bold cyan]Selective Restoration — {MODEL_NAME}[/bold cyan]\n"
        f"[yellow]Тестируем восстановление только верхних слоёв[/yellow]",
        title="🎯 Selective Restoration", box=box.DOUBLE_EDGE
    ))

    # Загружаем модель
    console.print("\n[yellow]Loading model (float16)...[/yellow]")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    model.eval()

    # Загружаем switches (top-100 на слой)
    with open("checkpoints/switches.json") as f:
        data = json.load(f)
    switches = {int(k): v[:100] for k, v in data["switches"].items()}

    # Сохраняем веса ВСЕХ модулей
    console.print("[cyan]Saving all module weights for switches...[/cyan]")
    saved = save_all_modules(model, switches)
    gc.collect()

    # Baseline
    console.print("[cyan]Baseline PPL...[/cyan]")
    base_ppl = compute_ppl(model, tok, TEST_PROMPTS)
    console.print(f"[green]Baseline: {base_ppl:.2f}[/green]")

    # Добавляем шум
    noise_std = 0.02
    console.print(f"[cyan]Adding noise ({noise_std})...[/cyan]")
    add_noise(model, noise_std)
    noisy_ppl = compute_ppl(model, tok, TEST_PROMPTS)
    console.print(f"[red]Noisy: {noisy_ppl:.2f}[/red]")

    # Тестируем selective restoration
    console.print("\n[cyan]Testing selective restoration...[/cyan]")
    results = []
    
    for start_layer in [0, 10, 15, 20, 22, 24, 26]:
        # Загружаем свежую модель
        model_test = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        model_test.eval()
        
        # Добавляем шум
        add_noise(model_test, noise_std)
        
        # Восстанавливаем
        count = restore_selective(model_test, saved, switches, start_layer)
        rest_ppl = compute_ppl(model_test, tok, TEST_PROMPTS)
        
        improvement = noisy_ppl / rest_ppl if rest_ppl > 0 else float('inf')
        results.append({
            "start_layer": start_layer,
            "restored_params": count,
            "restored_ppl": rest_ppl,
            "improvement": improvement
        })
        
        console.print(f"  Layer {start_layer:2d}+: {count:8,} params → PPL {rest_ppl:12.2f} ({improvement:8.1f}x)")
        
        del model_test
        gc.collect()

    # Таблица результатов
    table = Table(title="📊 Selective Restoration Results", box=box.ROUNDED)
    table.add_column("Start Layer", style="cyan", justify="right")
    table.add_column("Restored Params", style="magenta", justify="right")
    table.add_column("Restored PPL", style="green", justify="right")
    table.add_column("Improvement", style="yellow", justify="right")
    
    for r in results:
        table.add_row(
            f"{r['start_layer']}",
            f"{r['restored_params']:,}",
            f"{r['restored_ppl']:.2f}",
            f"{r['improvement']:.1f}x"
        )
    
    console.print("\n")
    console.print(table)

    # Сохраняем результаты
    output = {
        "model": MODEL_NAME,
        "noise_std": noise_std,
        "top_k_per_layer": 100,
        "baseline_ppl": base_ppl,
        "noisy_ppl": noisy_ppl,
        "selective_results": results,
        "timestamp": str(Path().cwd())
    }
    
    with open("checkpoints/selective_restoration.json", 'w') as f:
        json.dump(output, f, indent=2)
    
    console.print(f"\n✓ Результаты сохранены: checkpoints/selective_restoration.json")

if __name__ == "__main__":
    main()
