#!/usr/bin/env python3
"""
Генератор арифметических задач для эксперимента 02
RandomCalculation-style: синтетические примеры без утечек в pre-training
"""

import random
import json
from pathlib import Path
from typing import Tuple, List, Dict

class ArithmeticGenerator:
    def __init__(self, num_digits: int = 5, operations: List[str] = None):
        """
        num_digits: количество цифр в числах (5 = от 10000 до 99999)
        operations: список операций ['+', '-', '*']
        """
        self.num_digits = num_digits
        self.operations = operations or ['+', '-', '*']
    
    def generate(self) -> Tuple[str, int]:
        """Генерирует одну задачу: (prompt, answer)"""
        a = random.randint(10**(self.num_digits-1), 10**self.num_digits - 1)
        b = random.randint(10**(self.num_digits-1), 10**self.num_digits - 1)
        op = random.choice(self.operations)
        
        if op == '+':
            answer = a + b
        elif op == '-':
            # для вычитания гарантируем положительный ответ
            if a < b:
                a, b = b, a
            answer = a - b
        else:  # умножение
            # для умножения берём числа поменьше, чтобы ответ не взрывался
            a = random.randint(100, 999)   # 3 цифры
            b = random.randint(100, 999)   # 3 цифры
            answer = a * b
        
        prompt = f"Calculate: {a} {op} {b} = ?"
        return prompt, answer
    
    def generate_batch(self, n: int) -> List[Dict]:
        """Генерирует n задач и возвращает список словарей"""
        tasks = []
        for i in range(n):
            prompt, answer = self.generate()
            tasks.append({
                'id': i,
                'prompt': prompt,
                'answer': answer,
                'num_digits': self.num_digits,
                'operation': self.operations[0] if len(self.operations) == 1 else 'mixed'
            })
        return tasks
    
    def save_tasks(self, tasks: List[Dict], filepath: str):
        """Сохраняет задачи в JSON файл"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)


def main():
    """Тестовый запуск"""
    print("=" * 50)
    print("Генератор арифметических задач")
    print("=" * 50)
    
    # Генератор для 5-значных чисел
    gen = ArithmeticGenerator(num_digits=5, operations=['+', '-'])
    
    print("\nПримеры задач (5-значные числа, сложение и вычитание):")
    print("-" * 40)
    for i in range(10):
        prompt, answer = gen.generate()
        print(f"{i+1:2d}. {prompt} = {answer}")
    
    # Сохраняем тестовый набор
    tasks = gen.generate_batch(100)
    
    # Создаём папку для задач
    tasks_dir = Path(__file__).parent / "tasks"
    tasks_dir.mkdir(exist_ok=True)
    
    filepath = tasks_dir / f"arithmetic_{gen.num_digits}digit.json"
    gen.save_tasks(tasks, filepath)
    
    print(f"\n✅ Сохранено {len(tasks)} задач в: {filepath}")
    print(f"\nПример структуры:")
    print(json.dumps(tasks[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
