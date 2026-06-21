"""
Experiment 09 / Qwen2.5-1.5B — Switch Finder
Ищем переключателей по дисперсии активаций
"""
import json, warnings, logging
from datetime import datetime
from pathlib import Path
import numpy as np, torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
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

PROMPTS = [
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "Write a Python function to calculate the factorial of a number:",
    "Implement a binary search algorithm in Python:",
    "Write a function to check if a string is a palindrome:",
    "Create a class in Python that represents a bank account:",
    "What is 15 * 23?", "Solve for x: 2x + 5 = 17",
    "Calculate the derivative of f(x) = x^3 + 2x^2 - 5x + 3",
    "What is the integral of sin(x) dx?",
    "Find the sum of the first 100 natural numbers:",
    "Summarize: The quick brown fox jumps over the lazy dog.",
    "Summarize: Artificial intelligence is transforming various industries.",
    "Summarize: Climate change is a global challenge requiring immediate action.",
    "Summarize: The company reported record profits this quarter.",
    "Summarize: The new smartphone features an improved camera.",
    "If all roses are flowers and some flowers fade quickly, can we conclude some roses fade?",
    "A is taller than B. B is taller than C. Is A taller than C?",
    "If it rains, the ground gets wet. The ground is wet. Does it mean it rained?",
    "All cats are animals. Some animals are pets. Are all cats pets?",
    "If X > Y and Y > Z, is X > Z?",
    "Write a haiku about programming:", "Write a poem about the ocean:",
    "Describe a sunset in three sentences:",
    "What is the capital of France?", "Who wrote Romeo and Juliet?",
    "What is the speed of light?", "What is the largest planet?",
    "Translate to French: Hello, how are you?",
    "Translate to Spanish: Good morning!",
    "Translate to German: Thank you very much.",
    "Explain quantum computing in simple terms:",
    "What is machine learning? Explain like I'm 5:",
    "How does photosynthesis work?",
    "List 5 benefits of exercise:", "Name 3 programming languages for web:",
    "Compare Python and JavaScript:",
    "What's the difference between AI and machine learning?",
    "How to make a cup of coffee?", "How to tie a tie?",
    "Analyze the pros and cons of social media:",
    "What are the advantages of remote work?",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    return",
    "class Node:\n    def __init__(self, val):\n        self.val = val\n        self.next = None\n\ndef reverse(head):",
    "import math\ndef prime_factors(n):\n    factors = []\n    while n % 2 == 0:",
    "SELECT * FROM users WHERE age > 18 ORDER BY name;",
    "The mitochondria is the powerhouse of the cell. Explain why:",
    "What causes earthquakes? Explain the tectonic plate theory:",
    "Explain the water cycle in simple terms:",
    "What is DNA and how does it work?",
    "Who was Nikola Tesla and what were his contributions?",
    "Explain the theory of evolution by natural selection:",
    "What is the difference between a star and a planet?",
    "How does the internet work? Explain in simple terms:",
    "What is blockchain and how does it work?",
    "Explain the concept of supply and demand:",
    "What is inflation and how does it affect the economy?",
    "Explain the difference between stocks and bonds:",
    "What is GDP and why is it important?",
    "Explain how a refrigerator works:",
    "What is the difference between AC and DC current?",
    "How does a GPS system determine your location?",
    "Explain how a microwave oven heats food:",
    "What is the difference between RAM and ROM?",
    "How does Wi-Fi transmit data wirelessly?",
    "Explain the concept of recursion in programming:",
]


class ActivationTracker:
    def __init__(self):
        self.activations = {}
        self.hooks = []

    def hook_fn(self, layer_idx):
        def hook(module, input, output):
            h = output[0]
            if h.dim() == 3:
                last = h[:, -1, :]
            elif h.dim() == 2:
                last = h[-1, :]
            else:
                return
            self.activations[layer_idx] = last.squeeze().cpu().numpy()
        return hook

    def attach(self, model):
        for idx, layer in enumerate(model.model.layers):
            self.hooks.append(layer.register_forward_hook(self.hook_fn(idx)))

    def detach(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def main():
    console.print(Panel.fit(
        f"[bold cyan]Switch Finder — {MODEL_NAME}[/bold cyan]\n"
        f"Layers: {NUM_LAYERS} | Hidden: {HIDDEN_DIM} | Prompts: {len(PROMPTS)}",
        title="🔍 Phase 1", box=box.DOUBLE_EDGE
    ))

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    console.print("\n[yellow]Loading model...[/yellow]")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval()

    tracker = ActivationTracker()
    tracker.attach(model)

    all_acts = {i: [] for i in range(NUM_LAYERS)}
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  BarColumn(), TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task("Collecting activations", total=len(PROMPTS))
        for prompt in PROMPTS:
            tracker.activations = {}
            inputs = tok(prompt, return_tensors="pt")
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=5, do_sample=False)
            for layer_idx in range(NUM_LAYERS):
                if layer_idx in tracker.activations:
                    all_acts[layer_idx].append(tracker.activations[layer_idx])
            progress.update(task, advance=1)
    tracker.detach()

    console.print("\n[cyan]Finding switches...[/cyan]")
    switches = {}
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  BarColumn(), TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task("Analyzing layers", total=NUM_LAYERS)
        for layer_idx in range(NUM_LAYERS):
            acts = np.array(all_acts[layer_idx])
            layer_switches = []
            for n in range(acts.shape[1]):
                var = np.var(acts[:, n])
                mean = np.mean(np.abs(acts[:, n]))
                score = var / (mean + 1e-6)
                layer_switches.append((int(n), float(score)))
            layer_switches.sort(key=lambda x: x[1], reverse=True)
            switches[layer_idx] = layer_switches
            # Чекпоинт после каждого слоя
            with open(checkpoint_dir / f"switches_layer_{layer_idx}.json", 'w') as f:
                json.dump({"layer": layer_idx, "switches": layer_switches}, f)
            progress.update(task, advance=1)

    # Финальный чекпоинт
    with open(checkpoint_dir / "switches.json", 'w') as f:
        json.dump({
            "model": MODEL_NAME, "num_layers": NUM_LAYERS,
            "hidden_dim": HIDDEN_DIM, "n_prompts": len(PROMPTS),
            "switches": {str(k): v for k, v in switches.items()},
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)

    # Вывод топ-10
    table = Table(title=f"🎯 Top-10 Switches — {MODEL_NAME}", box=box.ROUNDED)
    table.add_column("Layer", style="cyan", justify="right")
    table.add_column("Neuron", style="magenta", justify="right")
    table.add_column("Score", style="green", justify="right")
    for li in sorted(switches.keys())[:5]:
        for ni, sc in switches[li][:10]:
            table.add_row(str(li), str(ni), f"{sc:.4f}")
    console.print("\n")
    console.print(table)
    console.print(f"\n[green]✓ Saved: checkpoints/switches.json[/green]")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--top_k", type=int, default=400)
    args = p.parse_args()
    main()
