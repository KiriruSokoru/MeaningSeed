#!/usr/bin/env python3
"""
MeaningSeed Surgery v1 - Germinate master neurons into a base model.
Applies seed (masters list) to a target model with surgical precision.
"""

import argparse
import json
import torch
import gc
from pathlib import Path
from typing import Dict, List, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.panel import Panel
from rich.progress import track, Progress
from rich.table import Table
from rich.syntax import Syntax

console = Console()


def load_seed(seed_path: str) -> Dict:
    """Load master neurons seed from JSON."""
    with open(seed_path, 'r') as f:
        seed = json.load(f)
    
    console.print(f"[green]Loaded seed: {seed_path}[/green]")
    console.print(f"  Version: {seed.get('version', 'unknown')}")
    console.print(f"  Model source: {seed.get('model', 'unknown')}")
    console.print(f"  Masters: {len(seed.get('masters', []))}")
    console.print(f"  Task: {seed.get('task', 'unknown')[:60]}...")
    
    return seed


def apply_surgery(
    model,
    masters: List[Dict],
    amplification: float = 1.3,
    attenuation: float = 0.7,
    dry_run: bool = False
) -> Tuple[Dict, Dict]:
    """
    Apply surgical modifications to model weights.
    
    Args:
        model: The target model
        masters: List of master neuron specs (layer_name, neuron_idx, mean, variance)
        amplification: Multiply master weights by this factor
        attenuation: Multiply non-master weights in same layer by this factor
        dry_run: If True, only report without modifying
    
    Returns:
        stats: Modification statistics
        backups: Original weights for rollback
    """
    stats = {
        'layers_modified': set(),
        'masters_applied': 0,
        'neurons_modified': 0,
        'total_weight_changes': 0
    }
    backups = {}
    
    # Group masters by layer
    masters_by_layer = {}
    for m in masters:
        layer_name = m['layer_name']
        if layer_name not in masters_by_layer:
            masters_by_layer[layer_name] = []
        masters_by_layer[layer_name].append(m['neuron_idx'])
    
    console.print(f"\n[bold]Surgery plan:[/bold] {len(masters_by_layer)} layers affected")
    
    for layer_name, neuron_indices in masters_by_layer.items():
        # Find the module
        module = model
        for part in layer_name.split('.'):
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part, None)
            if module is None:
                console.print(f"[red]Warning: Layer {layer_name} not found[/red]")
                break
        
        if module is None:
            continue
        
        # Get weight and bias
        if hasattr(module, 'weight'):
            weights = module.weight.data
            stats['layers_modified'].add(layer_name)
            
            if not dry_run:
                # Save backup
                backups[layer_name] = {
                    'weight': weights.clone(),
                    'bias': module.bias.data.clone() if hasattr(module, 'bias') and module.bias is not None else None
                }
                
                # Apply amplification to master neurons
                # For Linear layers: weight shape is [out_features, in_features]
                # Neuron index typically refers to output neuron
                for neuron_idx in neuron_indices:
                    if neuron_idx < weights.shape[0]:
                        weights[neuron_idx, :] *= amplification
                        stats['masters_applied'] += 1
                        stats['neurons_modified'] += 1
                        stats['total_weight_changes'] += weights.shape[1]
                
                # Apply attenuation to non-master neurons (optional)
                # This is commented by default - too aggressive
                # for i in range(weights.shape[0]):
                #     if i not in neuron_indices:
                #         weights[i, :] *= attenuation
                
                # Update bias if exists
                if hasattr(module, 'bias') and module.bias is not None:
                    for neuron_idx in neuron_indices:
                        if neuron_idx < module.bias.shape[0]:
                            module.bias.data[neuron_idx] *= amplification
        else:
            console.print(f"[dim]Layer {layer_name} has no weight, skipping[/dim]")
    
    return stats, backups


def validate_germination(model, tokenizer, task_prompt: str, device) -> Dict:
    """Test if the model produces valid JSON after surgery."""
    inputs = tokenizer(task_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Check for JSON
    import re
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    is_valid = False
    if json_match:
        try:
            json.loads(json_match.group())
            is_valid = True
        except:
            pass
    
    return {
        'response': response,
        'has_json': json_match is not None,
        'valid_json': is_valid,
        'json_length': len(json_match.group()) if json_match else 0
    }


def rollback(model, backups: Dict):
    """Restore original weights from backup."""
    console.print("[yellow]Rolling back changes...[/yellow]")
    for layer_name, backup in backups.items():
        module = model
        for part in layer_name.split('.'):
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part, None)
            if module is None:
                break
        
        if module and hasattr(module, 'weight'):
            module.weight.data = backup['weight']
            if backup['bias'] is not None and hasattr(module, 'bias') and module.bias is not None:
                module.bias.data = backup['bias']
    console.print("[green]Rollback complete[/green]")


def main():
    parser = argparse.ArgumentParser(description="MeaningSeed Surgery - Germinate masters into target model")
    parser.add_argument("--seed", type=str, required=True, help="Path to seed JSON file")
    parser.add_argument("--target", type=str, default="Qwen/Qwen2.5-0.5B", help="Target model")
    parser.add_argument("--amplify", type=float, default=1.3, help="Amplification factor for masters")
    parser.add_argument("--attenuate", type=float, default=0.7, help="Attenuation factor for non-masters")
    parser.add_argument("--test-prompt", type=str, default="Generate a JSON object with fields: name, age, city", help="Test prompt")
    parser.add_argument("--dry-run", action="store_true", help="Only report, don't modify")
    parser.add_argument("--rollback", action="store_true", help="Rollback previous surgery (requires backup)")
    args = parser.parse_args()
    
    console.print(Panel.fit(
        "[bold cyan]MeaningSeed Surgery v1[/bold cyan]\n"
        "Germinate master neurons into base model",
        border_style="cyan"
    ))
    
    # Load seed
    seed = load_seed(args.seed)
    masters = seed.get('masters', [])
    
    if not masters:
        console.print("[red]No masters found in seed[/red]")
        return
    
    # Load target model
    console.print(f"\n[dim]Loading target model: {args.target}[/dim]")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[dim]Device: {device}[/dim]")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.target,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Baseline test before surgery
    console.print("\n[bold]Baseline test (before surgery):[/bold]")
    baseline = validate_germination(model, tokenizer, args.test_prompt, device)
    console.print(f"  Valid JSON: {'✅' if baseline['valid_json'] else '❌'}")
    console.print(f"  Response preview: {baseline['response'][:200]}...")
    
    if args.rollback:
        # Rollback is not implemented for this session (would need persistent backups)
        console.print("[red]Rollback requires backup from previous surgery session[/red]")
        return
    
    # Apply surgery
    if args.dry_run:
        console.print("\n[yellow]DRY RUN - No modifications[/yellow]")
    else:
        console.print("\n[bold red]⚡ APPLYING SURGERY ⚡[/bold red]")
    
    stats, backups = apply_surgery(
        model, masters, 
        amplification=args.amplify,
        attenuation=args.attenuate,
        dry_run=args.dry_run
    )
    
    # Report stats
    table = Table(title="Surgery Report")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Layers modified", str(len(stats['layers_modified'])))
    table.add_row("Masters applied", str(stats['masters_applied']))
    table.add_row("Neurons modified", str(stats['neurons_modified']))
    table.add_row("Total weight changes", f"{stats['total_weight_changes']:,}")
    console.print(table)
    
    if args.dry_run:
        console.print("[yellow]Dry run complete. Run without --dry-run to apply.[/yellow]")
        return
    
    # Test after surgery
    console.print("\n[bold]Post-surgery test:[/bold]")
    post = validate_germination(model, tokenizer, args.test_prompt, device)
    
    console.print(f"  Valid JSON: {'✅' if post['valid_json'] else '❌'}")
    console.print(f"  Has JSON structure: {'✅' if post['has_json'] else '❌'}")
    
    console.print("\n[bold cyan]Response:[/bold cyan]")
    console.print(Syntax(post['response'], "json" if post['valid_json'] else "text", theme="monokai"))
    
    # Improvement check
    improved = post['valid_json'] and not baseline['valid_json']
    if improved:
        console.print("\n[green]✅ SUCCESS: Model learned to generate JSON![/green]")
    elif post['valid_json'] and baseline['valid_json']:
        console.print("\n[yellow]⚠️ Model already generated JSON (baseline)[/yellow]")
    else:
        console.print("\n[red]❌ No improvement. Try different seed or amplification factor.[/red]")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
