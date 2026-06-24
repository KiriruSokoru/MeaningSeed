#!/usr/bin/env python3
"""
Experiment 10: Baselines Comparison
ФИНАЛЬНАЯ ВЕРСИЯ:
- 1000 нейронов на слой (в модели 1536, берём top-1000)
- Без Fisher (убрано)
- Backup на диск
- TEST_SIZE=500, max_length=256
"""

import sys
import json
import time
import gc
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent / "09_switch_resonance" / "qwen_1.5b"))

console = Console()

# ============ КОНФИГУРАЦИЯ ============
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
NEURONS_PER_LAYER = 1000  # Исправлено: было 5000, в модели всего 1536
NOISE_LEVELS = [0.005, 0.02]
TEST_SIZE = 500
MAX_LENGTH = 256
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
BACKUP_DIR = CHECKPOINT_DIR / "backups"
BACKUP_DIR.mkdir(exist_ok=True)

# ============ УТИЛИТЫ ============

def save_checkpoint(name, data):
    path = CHECKPOINT_DIR / name
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    console.log(f"[green]✓ Чекпоинт:[/green] {path}")


def load_checkpoint(name):
    path = CHECKPOINT_DIR / name
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


def header(title):
    console.print()
    console.print(Panel(title, style="bold cyan"))
    console.print()


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============ ЗАГРУЗКА ДАННЫХ ============

def load_model_and_tokenizer():
    header("Загрузка модели")
    console.log(f"Model: {MODEL_NAME}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    model.eval()
    
    console.log(f"[green]✓ Модель загружена[/green]")
    return model, tokenizer


def load_test_data(tokenizer):
    header("Загрузка C4 датасета")
    console.log(f"Берём {TEST_SIZE} примеров")
    
    cached = load_checkpoint("test_data.json")
    if cached:
        console.log(f"[green]✓ Используем кешированные данные[/green]")
        texts = cached["texts"][:TEST_SIZE]
        prompts_sample = texts[:10]
        return texts, prompts_sample
    
    console.log("Загружаем C4 через streaming...")
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    
    texts = []
    for i, example in enumerate(dataset):
        if i >= TEST_SIZE:
            break
        text = example.get("text", "")
        if len(text.strip()) > 50:
            texts.append(text)
    
    console.log(f"[green]✓ Загружено {len(texts)} примеров[/green]")
    
    prompts_sample = texts[:10]
    console.print("\n[bold]Первые 10 промптов:[/bold]")
    for i, p in enumerate(prompts_sample, 1):
        preview = p[:100].replace("\n", " ")
        console.print(f"  {i}. {preview}...")
    
    save_checkpoint("test_data.json", {
        "texts": texts,
        "prompts_sample": prompts_sample,
        "count": len(texts)
    })
    
    return texts, prompts_sample


# ============ ПОИСК НЕЙРОНОВ ============

def find_switch_neurons(model, tokenizer, texts, neurons_per_layer):
    header("Поиск Switches")
    
    cached = load_checkpoint("neurons_switches.json")
    if cached and len(cached["neurons"]) == neurons_per_layer * 28:
        console.log(f"[green]✓ Используем кешированные switches[/green]")
        return cached["neurons"]
    
    console.log("Считаем variance активаций...")
    
    activations = {}
    hooks = []
    
    def make_hook(name):
        def hook(module, input, output):
            if name not in activations:
                activations[name] = []
            act = output.detach().cpu().float()
            activations[name].append(act.var(dim=(0, 1)).numpy())
        return hook
    
    for name, module in model.named_modules():
        if "mlp.down_proj" in name:
            hooks.append(module.register_forward_hook(make_hook(name)))
    
    console.log("Прогоняем 50 примеров...")
    sample_texts = texts[:50]
    
    for text in tqdm(sample_texts, desc="Сбор активаций"):
        inputs = tokenizer(text[:MAX_LENGTH], return_tensors="pt", truncation=True)
        with torch.no_grad():
            model(**inputs)
    
    for h in hooks:
        h.remove()
    
    neurons = []
    layer_idx = 0
    
    for name in sorted(activations.keys()):
        acts = np.array(activations[name])
        mean_var = acts.mean(axis=0)
        
        top_k = np.argsort(mean_var)[-neurons_per_layer:]
        
        for neuron_idx in top_k:
            neurons.append({
                "layer": layer_idx,
                "neuron": int(neuron_idx),
                "score": float(mean_var[neuron_idx]),
                "module": name
            })
        
        layer_idx += 1
    
    console.log(f"[green]✓ Найдено {len(neurons)} switches[/green]")
    
    save_checkpoint("neurons_switches.json", {"neurons": neurons})
    return neurons


def find_random_neurons(model, neurons_per_layer):
    header("Генерация Random нейронов")
    
    cached = load_checkpoint("neurons_random.json")
    if cached and len(cached["neurons"]) == neurons_per_layer * 28:
        console.log(f"[green]✓ Используем кешированные random[/green]")
        return cached["neurons"]
    
    n_layers = sum(1 for n, _ in model.named_modules() if "mlp.down_proj" in n)
    
    console.log(f"Генерируем {neurons_per_layer} случайных нейронов на {n_layers} слоёв...")
    
    neurons = []
    for layer_idx in range(n_layers):
        selected = random.sample(range(1536), neurons_per_layer)
        for neuron_idx in selected:
            neurons.append({
                "layer": layer_idx,
                "neuron": int(neuron_idx),
                "score": 0.0,
                "module": f"model.layers.{layer_idx}.mlp.down_proj"
            })
    
    console.log(f"[green]✓ Сгенерировано {len(neurons)} random нейронов[/green]")
    
    save_checkpoint("neurons_random.json", {"neurons": neurons})
    return neurons


def find_magnitude_neurons(model, neurons_per_layer):
    header("Поиск Magnitude нейронов")
    
    cached = load_checkpoint("neurons_magnitude.json")
    if cached and len(cached["neurons"]) == neurons_per_layer * 28:
        console.log(f"[green]✓ Используем кешированные magnitude[/green]")
        return cached["neurons"]
    
    console.log("Считаем magnitude...")
    
    neurons = []
    layer_idx = 0
    
    for name, module in model.named_modules():
        if "mlp.down_proj" not in name:
            continue
        
        weight = module.weight.detach().cpu().float()
        magnitudes = weight.abs().mean(dim=1).numpy()
        
        top_k = np.argsort(magnitudes)[-neurons_per_layer:]
        
        for neuron_idx in top_k:
            neurons.append({
                "layer": layer_idx,
                "neuron": int(neuron_idx),
                "score": float(magnitudes[neuron_idx]),
                "module": name
            })
        
        layer_idx += 1
    
    console.log(f"[green]✓ Найдено {len(neurons)} magnitude нейронов[/green]")
    
    save_checkpoint("neurons_magnitude.json", {"neurons": neurons})
    return neurons


# ============ ВОССТАНОВЛЕНИЕ И ОЦЕНКА ============

def save_backup_to_disk(model, noise_level):
    backup_path = BACKUP_DIR / f"backup_noise_{noise_level}.pt"
    
    if backup_path.exists():
        console.log(f"  [green]Backup уже на диске[/green]")
        return backup_path
    
    console.log(f"  Сохраняем backup на диск...")
    backup = {}
    for name, param in model.named_parameters():
        backup[name] = param.clone()
    
    torch.save(backup, backup_path)
    del backup
    cleanup_memory()
    
    return backup_path


def load_backup_from_disk(backup_path):
    return torch.load(backup_path, map_location="cpu")


def add_noise_inplace(model, noise_level):
    with torch.no_grad():
        for name, param in model.named_parameters():
            noise = torch.randn_like(param) * noise_level
            param.add_(noise)


def restore_from_backup(model, backup):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in backup:
                param.copy_(backup[name])


def compute_ppl(model, tokenizer, texts):
    model.eval()
    
    ppls = []
    total_nll = 0.0
    total_tokens = 0
    
    for text in tqdm(texts, desc="PPL", leave=False):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH
        )
        
        input_ids = inputs["input_ids"]
        if input_ids.shape[1] < 2:
            continue
        
        with torch.no_grad():
            outputs = model(**inputs, labels=input_ids)
            loss = outputs.loss.item()
        
        n_tokens = input_ids.shape[1] - 1
        nll = loss * n_tokens
        
        total_nll += nll
        total_tokens += n_tokens
        
        ppl = np.exp(loss)
        ppls.append(ppl)
    
    avg_ppl = np.exp(total_nll / total_tokens) if total_tokens > 0 else float('inf')
    
    return {
        "avg_ppl": float(avg_ppl),
        "per_prompt_ppl": [float(p) for p in ppls],
        "mean_prompt_ppl": float(np.mean(ppls)),
        "std_prompt_ppl": float(np.std(ppls)),
        "n_prompts": len(ppls)
    }


def run_restoration_test(model, tokenizer, texts, method_name, neurons, noise_level):
    checkpoint_name = f"ppl_{method_name}_{noise_level}.json"
    
    cached = load_checkpoint(checkpoint_name)
    if cached:
        console.log(f"[green]✓ Кешированный PPL для {method_name} @ {noise_level}[/green]")
        return cached
    
    console.log(f"Тестируем {method_name} @ noise={noise_level}...")
    
    backup_path = save_backup_to_disk(model, noise_level)
    add_noise_inplace(model, noise_level)
    
    backup = load_backup_from_disk(backup_path)
    
    by_module = {}
    for n in neurons:
        mod = n["module"]
        if mod not in by_module:
            by_module[mod] = []
        by_module[mod].append(n["neuron"])
    
    with torch.no_grad():
        for mod_name, neuron_indices in by_module.items():
            weight_name = f"{mod_name}.weight"
            for name, param in model.named_parameters():
                if name == weight_name:
                    param[neuron_indices, :] = backup[name][neuron_indices, :]
                    break
            
            bias_name = f"{mod_name}.bias"
            for name, param in model.named_parameters():
                if name == bias_name:
                    param[neuron_indices] = backup[name][neuron_indices]
                    break
    
    console.log(f"  Считаем PPL...")
    ppl_result = compute_ppl(model, tokenizer, texts)
    
    result = {
        "method": method_name,
        "noise_level": noise_level,
        **ppl_result
    }
    
    save_checkpoint(checkpoint_name, result)
    
    restore_from_backup(model, backup)
    
    del backup
    cleanup_memory()
    
    return result


# ============ ВИЗУАЛИЗАЦИЯ ============

def show_results_table(results, baseline_ppl):
    header("РЕЗУЛЬТАТЫ")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Method", style="cyan")
    table.add_column("Noise", style="green")
    table.add_column("PPL", style="yellow")
    table.add_column("vs Baseline", style="blue")
    table.add_column("Improvement", style="red")
    
    for r in results:
        noise = r["noise_level"]
        ppl = r["avg_ppl"]
        
        noisy_ppl = next((x["avg_ppl"] for x in results if x["method"] == "noisy" and x["noise_level"] == noise), None)
        
        vs_baseline = f"{ppl/baseline_ppl:.2f}x" if baseline_ppl > 0 else "N/A"
        
        if noisy_ppl and noisy_ppl > 0:
            improvement = f"{noisy_ppl/ppl:.2f}x"
        else:
            improvement = "N/A"
        
        table.add_row(
            r["method"],
            str(noise),
            f"{ppl:.2f}",
            vs_baseline,
            improvement
        )
    
    console.print(table)


def plot_results(results, baseline_ppl):
    header("ГРАФИКИ")
    
    methods = ["switches", "random", "magnitude"]
    colors = {"switches": "#2ecc71", "random": "#e74c3c", "magnitude": "#3498db"}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    noise = 0.02
    method_ppls = []
    method_names = []
    method_colors = []
    
    for method in methods:
        r = next((x for x in results if x["method"] == method and x["noise_level"] == noise), None)
        if r:
            method_ppls.append(r["avg_ppl"])
            method_names.append(method)
            method_colors.append(colors[method])
    
    noisy_r = next((x for x in results if x["method"] == "noisy" and x["noise_level"] == noise), None)
    if noisy_r:
        method_ppls.insert(0, noisy_r["avg_ppl"])
        method_names.insert(0, "noisy")
        method_colors.insert(0, "#95a5a6")
    
    method_ppls.insert(0, baseline_ppl)
    method_names.insert(0, "baseline")
    method_colors.insert(0, "#27ae60")
    
    ax.bar(method_names, method_ppls, color=method_colors)
    ax.set_ylabel("Perplexity (lower = better)")
    ax.set_title(f"PPL comparison @ noise={noise}")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plot_path = CHECKPOINT_DIR / "ppl_comparison_bar.png"
    plt.savefig(plot_path, dpi=150)
    console.log(f"[green]✓ График:[/green] {plot_path}")
    plt.close()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method in methods:
        xs = []
        ys = []
        for noise in sorted(NOISE_LEVELS):
            r = next((x for x in results if x["method"] == method and x["noise_level"] == noise), None)
            if r:
                xs.append(noise)
                ys.append(r["avg_ppl"])
        if xs:
            ax.plot(xs, ys, marker="o", label=method, color=colors[method], linewidth=2)
    
    ax.axhline(y=baseline_ppl, color="green", linestyle="--", label="baseline")
    for noise in NOISE_LEVELS:
        noisy_r = next((x for x in results if x["method"] == "noisy" and x["noise_level"] == noise), None)
        if noisy_r:
            ax.plot(noise, noisy_r["avg_ppl"], "x", color="gray", markersize=10)
    
    ax.set_xlabel("Noise level")
    ax.set_ylabel("Perplexity")
    ax.set_title("PPL vs Noise")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = CHECKPOINT_DIR / "ppl_vs_noise.png"
    plt.savefig(plot_path, dpi=150)
    console.log(f"[green]✓ График:[/green] {plot_path}")
    plt.close()


# ============ MAIN ============

def main():
    console.print()
    console.print(Panel("Experiment 10: Baselines Comparison (FINAL)", style="bold cyan"))
    console.print(f"Model: {MODEL_NAME}")
    console.print(f"Neurons per layer: {NEURONS_PER_LAYER} (из 1536 доступных)")
    console.print(f"Noise levels: {NOISE_LEVELS}")
    console.print(f"Test size: {TEST_SIZE}")
    console.print(f"Max length: {MAX_LENGTH}")
    console.print(f"Methods: switches, random, magnitude (без fisher)")
    
    start_time = time.time()
    
    model, tokenizer = load_model_and_tokenizer()
    texts, prompts_sample = load_test_data(tokenizer)
    
    # Baseline
    header("Baseline PPL")
    cached_baseline = load_checkpoint("ppl_baseline.json")
    if cached_baseline:
        baseline_ppl = cached_baseline["avg_ppl"]
        console.log(f"[green]✓ Baseline PPL: {baseline_ppl:.2f}[/green]")
    else:
        baseline_result = compute_ppl(model, tokenizer, texts)
        baseline_ppl = baseline_result["avg_ppl"]
        save_checkpoint("ppl_baseline.json", {"avg_ppl": baseline_ppl, **baseline_result})
        console.log(f"Baseline PPL: {baseline_ppl:.2f}")
    
    # Noisy PPL
    for noise in NOISE_LEVELS:
        checkpoint_name = f"ppl_noisy_{noise}.json"
        cached = load_checkpoint(checkpoint_name)
        if cached:
            console.log(f"[green]✓ Noisy PPL @{noise}: {cached['avg_ppl']:.2f}[/green]")
            continue
        
        header(f"Noisy PPL @ noise={noise}")
        backup_path = save_backup_to_disk(model, noise)
        add_noise_inplace(model, noise)
        noisy_result = compute_ppl(model, tokenizer, texts)
        save_checkpoint(checkpoint_name, {"avg_ppl": noisy_result["avg_ppl"], "noise_level": noise, **noisy_result})
        console.log(f"Noisy PPL @{noise}: {noisy_result['avg_ppl']:.2f}")
        
        backup = load_backup_from_disk(backup_path)
        restore_from_backup(model, backup)
        del backup
        cleanup_memory()
    
    # Нейроны (без Fisher)
    all_neurons = {}
    all_neurons["switches"] = find_switch_neurons(model, tokenizer, texts, NEURONS_PER_LAYER)
    all_neurons["random"] = find_random_neurons(model, NEURONS_PER_LAYER)
    all_neurons["magnitude"] = find_magnitude_neurons(model, NEURONS_PER_LAYER)
    
    # Тесты
    all_results = []
    all_results.append({"method": "baseline", "noise_level": 0, "avg_ppl": baseline_ppl})
    for noise in NOISE_LEVELS:
        cached = load_checkpoint(f"ppl_noisy_{noise}.json")
        if cached:
            all_results.append(cached)
    
    for method_name, neurons in all_neurons.items():
        for noise in NOISE_LEVELS:
            result = run_restoration_test(model, tokenizer, texts, method_name, neurons, noise)
            all_results.append(result)
            console.log(f"  [cyan]{method_name}[/cyan] @{noise}: PPL = {result['avg_ppl']:.2f}")
    
    # Финал
    save_checkpoint("final_results.json", {
        "baseline_ppl": baseline_ppl,
        "results": all_results,
        "config": {
            "model": MODEL_NAME,
            "neurons_per_layer": NEURONS_PER_LAYER,
            "noise_levels": NOISE_LEVELS,
            "test_size": TEST_SIZE,
            "methods": ["switches", "random", "magnitude"]
        }
    })
    
    show_results_table(all_results, baseline_ppl)
    plot_results(all_results, baseline_ppl)
    
    header("АНАЛИЗ")
    
    for noise in NOISE_LEVELS:
        console.print(f"\n[bold]Noise = {noise}:[/bold]")
        
        best_method = None
        best_ppl = float("inf")
        
        for method in ["switches", "random", "magnitude"]:
            r = next((x for x in all_results if x["method"] == method and x["noise_level"] == noise), None)
            if r and r["avg_ppl"] < best_ppl:
                best_ppl = r["avg_ppl"]
                best_method = method
        
        console.print(f"  Лучший: [green]{best_method}[/green] (PPL = {best_ppl:.2f})")
        
        if best_method == "switches":
            console.print("  [green]✅ SUCCESS: Switches побеждают![/green]")
        elif best_method == "magnitude":
            console.print("  [yellow]⚠️  Magnitude лучше switches[/yellow]")
        else:
            console.print("  [red]❌ Random лучший — что-то не так[/red]")
    
    elapsed = time.time() - start_time
    console.print(f"\n[bold]Время:[/bold] {elapsed/60:.1f} минут")
    console.print(f"[bold]Результаты:[/bold] {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
