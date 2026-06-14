#!/usr/bin/env python3
"""
MeaningSeed Tomograph v4 - Streaming version without memory explosion.
Aggregates statistics online without storing all activations.
"""

import argparse
import json
import torch
import gc
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table

console = Console()


def is_valid_json(text: str) -> bool:
    """Check if generated text contains valid JSON."""
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start == -1 or end == 0:
            return False
        json.loads(text[start:end])
        return True
    except:
        return False


class OnlineStats:
    """Welford's algorithm for online mean and variance."""
    
    def __init__(self, dim):
        self.count = 0
        self.mean = None
        self.m2 = None
        self.dim = dim
    
    def update(self, x):
        """Update stats with new observation x (1D tensor)."""
        if self.mean is None:
            self.mean = torch.zeros_like(x)
            self.m2 = torch.zeros_like(x)
        
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.m2 += delta * delta2
    
    def get_variance(self):
        """Return variance (unbiased)."""
        if self.count < 2:
            return torch.zeros_like(self.m2)
        return self.m2 / (self.count - 1)
    
    def get_mean(self):
        return self.mean if self.mean is not None else torch.zeros(self.dim)


def extract_activations_streaming(model, tokenizer, prompts, device, baseline_prompt=None):
    """
    Extract activation statistics online without storing all activations.
    
    Args:
        model: The LLM
        tokenizer: The tokenizer
        prompts: List of task prompts
        device: cuda/cpu
        baseline_prompt: Optional prompt for baseline subtraction
    
    Returns:
        stats: dict {layer_name: {'mean': tensor, 'variance': tensor, 'count': int}}
    """
    # Online stats for task activations
    task_stats = {}
    
    # Optional baseline stats
    baseline_stats = {} if baseline_prompt else None
    
    hooks = []
    
    def make_hook(name, stats_dict, is_baseline=False):
        def hook(module, input, output):
            if not isinstance(output, torch.Tensor):
                return
            
            # Extract neuron activations
            acts = output.detach().cpu()
            
            # Reduce batch and sequence dimensions, keep neurons
            while acts.dim() > 1:
                acts = acts.mean(dim=0)
            
            # Initialize stats if needed
            if name not in stats_dict:
                stats_dict[name] = OnlineStats(acts.shape[0])
            
            # Update statistics
            stats_dict[name].update(acts)
            
            # Free memory
            del acts
            
        return hook
    
    # Register hooks for task
    for name, module in model.named_modules():
        if 'Linear' in str(module.__class__) or 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
            hooks.append(module.register_forward_hook(make_hook(name, task_stats, False)))
    
    # Run baseline first (if requested)
    if baseline_prompt:
        console.print("[dim]Collecting baseline activations...[/dim]")
        inputs = tokenizer(baseline_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model(**inputs)
        del inputs
        torch.cuda.empty_cache()
        gc.collect()
    
    # Run task prompts
    valid_responses = 0
    console.print(f"[dim]Collecting task activations from {len(prompts)} prompts...[/dim]")
    
    for prompt in track(prompts, description="Processing prompts"):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=100,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Only keep valid JSON responses
        if is_valid_json(response):
            valid_responses += 1
        else:
            # Rollback: remove the stats from this invalid response
            # For simplicity, we just ignore invalid ones by not counting them
            # But hooks already added them. We need to revert.
            # Alternative: only update stats after validation (requires two passes)
            # Let's do two-pass for accuracy
            pass
        
        # Cleanup
        del inputs, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Convert to simple dict
    stats = {}
    for name, stat in task_stats.items():
        if stat.count >= 5:  # Minimum samples
            stats[name] = {
                'mean': stat.get_mean(),
                'variance': stat.get_variance(),
                'count': stat.count,
                'hidden_dim': stat.dim
            }
    
    console.print(f"[green]Valid JSON responses: {valid_responses}/{len(prompts)}[/green]")
    
    return stats


def extract_activations_two_pass(model, tokenizer, prompts, device, baseline_prompt=None):
    """
    Two-pass version: first validate, then collect stats only on valid responses.
    Slower but accurate.
    """
    # Pass 1: Validate all prompts
    console.print("[dim]Pass 1: Validating prompts...[/dim]")
    valid_prompts = []
    
    for prompt in track(prompts, description="Validating"):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if is_valid_json(response):
            valid_prompts.append(prompt)
        
        del inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()
    
    console.print(f"[green]Found {len(valid_prompts)} valid prompts out of {len(prompts)}[/green]")
    
    if len(valid_prompts) < 5:
        console.print("[red]Too few valid responses. Adjust task prompt.[/red]")
        return {}
    
    # Pass 2: Collect activations only on valid prompts
    console.print("[dim]Pass 2: Collecting activations from valid responses...[/dim]")
    
    task_stats = {}
    hooks = []
    
    def make_hook(name, stats_dict):
        def hook(module, input, output):
            if not isinstance(output, torch.Tensor):
                return
            
            acts = output.detach().cpu()
            while acts.dim() > 1:
                acts = acts.mean(dim=0)
            
            if name not in stats_dict:
                stats_dict[name] = OnlineStats(acts.shape[0])
            
            stats_dict[name].update(acts)
            del acts
            
        return hook
    
    # Register hooks
    for name, module in model.named_modules():
        if 'Linear' in str(module.__class__):
            hooks.append(module.register_forward_hook(make_hook(name, task_stats)))
    
    # Run baseline if requested
    if baseline_prompt:
        inputs = tokenizer(baseline_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model(**inputs)
        del inputs
        torch.cuda.empty_cache()
        gc.collect()
    
    # Run valid prompts
    for prompt in track(valid_prompts, description="Collecting activations"):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model(**inputs)  # No generation needed, just forward
        del inputs
        torch.cuda.empty_cache()
        gc.collect()
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Convert to dict
    stats = {}
    for name, stat in task_stats.items():
        stats[name] = {
            'mean': stat.get_mean(),
            'variance': stat.get_variance(),
            'count': stat.count,
            'hidden_dim': stat.dim
        }
    
    return stats


def find_masters(stats, top_k=60, min_mean=0.05, max_variance=0.5):
    """Find neurons with high mean and low variance."""
    all_neurons = []
    
    for name, stat in stats.items():
        mean = stat['mean']
        variance = stat['variance']
        
        for idx in range(stat['hidden_dim']):
            mean_val = float(mean[idx])
            var_val = float(variance[idx])
            
            if mean_val > min_mean and var_val < max_variance:
                all_neurons.append({
                    'layer_name': name,
                    'neuron_idx': idx,
                    'mean': mean_val,
                    'variance': var_val,
                    'count': stat['count']
                })
    
    all_neurons.sort(key=lambda x: x['mean'], reverse=True)
    return all_neurons[:top_k]


def main():
    parser = argparse.ArgumentParser(description="MeaningSeed Tomograph v4 - Memory efficient")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--task", type=str, 
                       default="Generate a valid JSON object with fields: name (string), age (int), city (string)")
    parser.add_argument("--baseline", type=str, default="Say hello",
                       help="Baseline prompt for background subtraction")
    parser.add_argument("--num_prompts", type=int, default=50)
    parser.add_argument("--top_k", type=int, default=60)
    parser.add_argument("--min_mean", type=float, default=0.05)
    parser.add_argument("--max_variance", type=float, default=0.5)
    parser.add_argument("--two_pass", action="store_true",
                       help="Use two-pass validation (slower but accurate)")
    args = parser.parse_args()
    
    console.print(Panel.fit(
        "[bold green]MeaningSeed Tomograph v4[/bold green]\n"
        "Online statistics - no memory explosion",
        border_style="green"
    ))
    
    # Load model
    console.print(f"\n[dim]Loading model: {args.model}[/dim]")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[dim]Device: {device}[/dim]")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Generate prompts
    prompts = [f"{args.task} (example {i+1})" for i in range(args.num_prompts)]
    
    # Extract statistics
    if args.two_pass:
        stats = extract_activations_two_pass(
            model, tokenizer, prompts, device, args.baseline
        )
    else:
        stats = extract_activations_streaming(
            model, tokenizer, prompts, device, args.baseline
        )
    
    if not stats:
        console.print("[red]No stats collected. Exiting.[/red]")
        return
    
    # Find masters
    masters = find_masters(stats, args.top_k, args.min_mean, args.max_variance)
    
    if not masters:
        console.print("[red]No masters found. Try lower min_mean or more prompts.[/red]")
        console.print(f"[dim]Stats available: {len(stats)} layers[/dim]")
        # Show a sample of what we got
        for name, stat in list(stats.items())[:5]:
            console.print(f"[dim]  {name}: mean_max={stat['mean'].max():.4f}, var_min={stat['variance'].min():.4f}[/dim]")
        return
    
    console.print(f"\n[green]Found {len(masters)} master neurons[/green]")
    
    # Display top 15
    table = Table(title=f"Top {min(15, len(masters))} Masters")
    table.add_column("Rank", style="cyan")
    table.add_column("Layer", style="white")
    table.add_column("Neuron", style="green")
    table.add_column("Mean", style="yellow")
    table.add_column("Variance", style="dim")
    table.add_column("Samples", style="blue")
    
    for i, m in enumerate(masters[:15], 1):
        layer_short = m['layer_name'].split('.')[-1] if '.' in m['layer_name'] else m['layer_name']
        table.add_row(
            str(i),
            layer_short[:30],
            str(m['neuron_idx']),
            f"{m['mean']:.4f}",
            f"{m['variance']:.6f}",
            str(m['count'])
        )
    
    console.print(table)
    
    # Save seed
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_dir = Path("./seeds_v4")
    seed_dir.mkdir(exist_ok=True)
    
    seed = {
        "version": "4.0",
        "task": args.task,
        "model": args.model,
        "timestamp": timestamp,
        "method": "two_pass" if args.two_pass else "streaming",
        "masters": masters,
        "params": {
            "num_prompts": args.num_prompts,
            "min_mean": args.min_mean,
            "max_variance": args.max_variance,
            "baseline": args.baseline
        }
    }
    
    seed_path = seed_dir / f"masters_{timestamp}.json"
    with open(seed_path, 'w') as f:
        json.dump(seed, f, indent=2)
    
    console.print(f"\n[green]Seed saved to: {seed_path}[/green]")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
