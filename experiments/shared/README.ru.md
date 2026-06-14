# Общие скрипты для экспериментов MeaningSeed

## tomograph_v4.py
Диагностика мастер-нейронов. Находит нейроны с высокой средней активацией и низкой дисперсией.

Использование:
```bash
python tomograph_v4.py --model MODEL --task "TASK" --num_prompts N

surgery_v1.py

Проращивание семени в целевую модель. Усиливает мастер-нейроны из seed.

Использование:
bash

python surgery_v1.py --seed PATH --target MODEL --amplify 1.3

