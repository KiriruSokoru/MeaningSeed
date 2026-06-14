# Seeds

В этой папке хранятся семена (мастер-нейроны), извлечённые из LLM.

## Формат семени

```json
{
  "version": "4.0",
  "model_source": "Qwen/Qwen2.5-0.5B-Instruct",
  "task": "описание задачи",
  "masters": [
    {
      "layer": "model.layers.0.self_attn.k_proj",
      "neuron": 31,
      "mean": 120.96,
      "variance": 0.082
    }
  ]
}

Список семян
Файл	Задача	Источник
qwen2_0.5b_poc_v1_layer*.json	POC	Qwen2-0.5B
Примечание

Сами файлы .json не хранятся в репозитории (через .gitignore),
но структура папки и этот README сохраняются.
