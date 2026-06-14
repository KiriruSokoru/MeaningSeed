import json
from pathlib import Path

# Загружаем исходные задачи
with open("tasks/arithmetic_5digit.json") as f:
    tasks = json.load(f)

# Конвертируем в Qwen Instruct формат
fixed_tasks = []
for task in tasks:
    prompt = task['prompt']
    # prompt вида "Calculate: 98947 - 71228 = ?"
    # Вытаскиваем выражение
    expr = prompt.replace("Calculate:", "").replace("= ?", "").strip()
    
    # Новый промпт в формате Qwen
    new_prompt = f"<|im_start|>user\nCompute {expr}\n<|im_end|>\n<|im_start|>assistant\nThe answer is"
    
    fixed_tasks.append({
        'id': task['id'],
        'original_prompt': task['prompt'],
        'prompt': new_prompt,
        'answer': task['answer']
    })

# Сохраняем
with open("tasks/arithmetic_5digit_fixed.json", "w") as f:
    json.dump(fixed_tasks, f, indent=2)

print(f"Сконвертировано {len(fixed_tasks)} задач")
print("\nПример:")
print(fixed_tasks[0]['prompt'])
