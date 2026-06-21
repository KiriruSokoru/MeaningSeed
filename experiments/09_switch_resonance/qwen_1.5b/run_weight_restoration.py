"""
Experiment 09 / Qwen2.5-1.5B — Weight Restoration (Ultra Optimized)
"""
import json, warnings, logging, gc
from datetime import datetime
from pathlib import Path
import torch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
console = Console()

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
HIDDEN_DIM = 1536
NUM_LAYERS = 28

INPUT_MODULES = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
                 'mlp.gate_proj', 'mlp.up_proj']
OUTPUT_MODULES = ['self_attn.o_proj', 'mlp.down_proj']

TEST_PROMPTS = [
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "Calculate: 24578 + 13892 =",
    "The capital of France is",
    "Write a Python function to calculate factorial:",
    "Summarize: Artificial intelligence is transforming industries.",
]


def add_noise_blockwise(model, noise_std, block_size_mb=10):
    """Добавляет шум по блокам, чтобы не OOM"""
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                # Вычисляем размер блока
                total_elements = p.numel()
                block_elements = int((block_size_mb * 1024 * 1024) / 4)  # float32 = 4 bytes
                block_elements = max(1, min(block_elements, total_elements))
                
                # Обрабатываем по блокам
                for start in range(0, total_elements, block_elements):
                    end = min(start + block_elements, total_elements)
                    flat = p.view(-1)
                    noise = torch.randn(end - start, device=p.device, dtype=p.dtype) * noise_std
                    flat[start:end].add_(noise)
                    del noise
                p.data = flat.view_as(p)
    gc.collect()


def save_switch_weights_only(model, switches):
    """Сохраняет ТОЛЬКО веса переключателей, а не все"""
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
    gc.collect()
    return saved


def restore_switches(model, saved, switches):
    """Восстанавливает переключателей из сохранённых"""
    count = 0
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
                        count += 1
                for mod_name in OUTPUT_MODULES:
                    mod = layer
                    for part in mod_name.split('.'):
                        mod = getattr(mod, part)
                    w = mod.weight
                    key = f"model.layers.{layer_idx}.{mod_name}.weight"
                    if key in saved and neuron_idx in saved[key]:
                        w.data[neuron_idx, :] = saved[key][neuron_idx].to(w.device)
                        count += 1
    gc.collect()
    return count


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
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--noise_std", type=float, default=0.005)
    p.add_argument("--top_k", type=int, default=400)
    args = p.parse_args()

    console.print(Panel.fit(
        f"[bold cyan]Weight Restoration — {MODEL_NAME}[/bold cyan]\n"
        f"Noise: {args.noise_std} | Top-K: {args.top_k}\n"
        f"[yellow]Ultra optimized: float16 + blockwise noise + switch-only save[/yellow]",
        title="🔧 Phase 4 (Ultra Optimized)", box=box.DOUBLE_EDGE
    ))

    ckpt = Path("checkpoints")
    with open(ckpt / "switches.json") as f:
        data = json.load(f)
    switches = {}
    for k, v in data["switches"].items():
        switches[int(k)] = v[:args.top_k]

    console.print("\n[yellow]Loading model (float16)...[/yellow]")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Загружаем в float16 для экономии памяти
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16,  # ← float16 вместо float32
        low_cpu_mem_usage=True
    )
    model.eval()

    console.print("[cyan]Saving switch weights only...[/cyan]")
    saved = save_switch_weights_only(model, switches)
    gc.collect()

    console.print("[cyan]Baseline PPL...[/cyan]")
    base_ppl = compute_ppl(model, tok, TEST_PROMPTS)
    console.print(f"[green]Baseline: {base_ppl:.2f}[/green]")

    console.print(f"[cyan]Adding noise ({args.noise_std}) blockwise...[/cyan]")
    add_noise_blockwise(model, args.noise_std, block_size_mb=5)
    
    console.print("[cyan]Noisy PPL...[/cyan]")
    noisy_ppl = compute_ppl(model, tok, TEST_PROMPTS)
    console.print(f"[red]Noisy: {noisy_ppl:.2f}[/red]")

    console.print("[cyan]Restoring switches...[/cyan]")
    count = restore_switches(model, saved, switches)
    
    console.print("[cyan]Restoration PPL...[/cyan]")
    rest_ppl = compute_ppl(model, tok, TEST_PROMPTS)
    console.print(f"[green]Restored: {rest_ppl:.2f} ({count} params)[/green]")

    del saved
    gc.collect()

    results = {
        "model": MODEL_NAME,
        "noise_std": args.noise_std,
        "top_k": args.top_k,
        "baseline_ppl": base_ppl,
        "noisy_ppl": noisy_ppl,
        "restoration_ppl": rest_ppl,
        "improvement_factor": noisy_ppl / rest_ppl if rest_ppl > 0 else float('inf'),
        "restored_params": count,
        "timestamp": datetime.now().isoformat()
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(ckpt / f"restoration_{ts}.json", 'w') as f:
        json.dump(results, f, indent=2)

    table = Table(title="📊 Results", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta", justify="right")
    table.add_row("Baseline PPL", f"{base_ppl:.2f}")
    table.add_row("Noisy PPL", f"{noisy_ppl:.2f}")
    table.add_row("Restoration PPL", f"{rest_ppl:.2f}")
    table.add_row("Improvement", f"{results['improvement_factor']:.1f}x")
    table.add_row("Restored params", f"{count}")
    console.print("\n")
    console.print(table)


if __name__ == "__main__":
    main()
