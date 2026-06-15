import json
import logging
import os
from typing import Dict, List, Any, Optional, Tuple

from .i18n import get_t

__all__ = [
    "load_seed",
    "validate_seed_compatibility",
    "save_seed",
    "SeedRegistry"
]

logger = logging.getLogger(__name__)


def load_seed(seed_path: str) -> Dict[str, Any]:
    """
    Загрузить и распарсить JSON-файл семени.
    
    Args:
        seed_path: Путь к JSON-файлу с семенем
        
    Returns:
        Словарь с данными семени
        
    Raises:
        FileNotFoundError: Если файл не существует
        json.JSONDecodeError: Если файл содержит невалидный JSON
        
    Example:
        >>> seed_data = load_seed("./seeds/honesty_seed.json")
        >>> print(seed_data["model_type"])
        "qwen2"
    """
    with open(seed_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def validate_seed_compatibility(
    seed_data: Dict[str, Any],
    model_config: Any
) -> Tuple[bool, str]:
    """
    Проверить совместимость семени с конфигурацией модели до применения.
    
    Выполняет строгую проверку совместимости семени с текущей моделью:
    1. Проверка совпадения типа архитектуры (model_type)
    2. Проверка совпадения размера скрытого слоя (hidden_size)
    
    Args:
        seed_data: Словарь с данными семени из JSON-файла
        model_config: Конфигурация модели (transformers PretrainedConfig)
        
    Returns:
        Кортеж из двух элементов:
        - is_compatible (bool): True если семя совместимо с моделью
        - error_message (str): Сообщение об ошибке или подтверждение совместимости
        
    Example:
        >>> is_compat, msg = validate_seed_compatibility(seed_data, model.config)
        >>> if not is_compat:
        ...     print(f"Ошибка: {msg}")
    """
    seed_type = seed_data.get("model_type")
    config_type = getattr(model_config, "model_type", "unknown")

    if seed_type and config_type and seed_type != config_type:
        return False, (
            f"Несовпадение типа модели: Сид создан для '{seed_type}', "
            f"но загружена модель '{config_type}'."
        )

    seed_hidden_size = seed_data.get("model_hidden_size")
    config_hidden_size = getattr(model_config, "hidden_size", None)

    if seed_hidden_size and config_hidden_size and seed_hidden_size != config_hidden_size:
        return False, (
            f"Критическое несовпадение размерностей: Сид ожидает hidden_size={seed_hidden_size} "
            f"(например, 0.5B), но у загруженной модели hidden_size={config_hidden_size} "
            f"(например, 1.5B). Применение весов невозможно без искажения тензоров."
        )

    return True, "Совместимость подтверждена"


def save_seed(seed_data: Dict[str, Any], seed_path: str) -> None:
    """
    Сохранить данные семени в JSON-файл с обновлёнными метаданными.
    
    Args:
        seed_data: Словарь с данными семени для сохранения
        seed_path: Путь к файлу для сохранения семени
        
    Raises:
        OSError: Если не удалось записать файл (нет прав доступа, диск заполнен)
        
    Example:
        >>> seed_data = {
        ...     "model_type": "qwen2",
        ...     "layer_idx": 12,
        ...     "master_indices": [100, 200, 300],
        ...     "scale": 1.5
        ... }
        >>> save_seed(seed_data, "./seeds/my_seed.json")
    """
    with open(seed_path, 'w', encoding='utf-8') as f:
        json.dump(seed_data, f, indent=2, ensure_ascii=False)


class SeedRegistry:
    """
    Реестр топологических семян (proofs) с кэшированием и валидацией.
    
    Класс управляет коллекцией JSON-файлов с семенами в указанной директории:
    - Загрузка и кэширование семян из файлов
    - Регистрация новых семян (proofs)
    - Поиск семян по имени
    - Проверка совместимости семян с моделями
    
    Реестр использует ленивую загрузку: файлы не читаются при инициализации.
    Для загрузки семян из директории необходимо явно вызвать метод refresh_cache().
    
    Attributes:
        registry_dir: Путь к директории с JSON-файлами семян
        lang: Язык для сообщений ("ru" или "en")
        
    Example:
        >>> registry = SeedRegistry("./seeds", lang="ru")
        >>> registry.refresh_cache()  # Явная загрузка семян из файлов
        >>> proofs = registry.list_proofs()
        >>> print(proofs)
        ['honesty_seed', 'safety_seed', 'creativity_seed']
    """

    def __init__(self, registry_dir: str = "./seeds", lang: str = "ru") -> None:
        """
        Инициализировать реестр семян.
        
        Создаёт директорию для хранения семян (если не существует),
        но НЕ загружает файлы из неё. Для загрузки используйте refresh_cache().
        
        Args:
            registry_dir: Путь к директории с JSON-файлами семян
            lang: Язык для сообщений ("ru" или "en")
        """
        self.registry_dir: str = registry_dir
        self.lang: str = lang
        os.makedirs(self.registry_dir, exist_ok=True)
        self._cache: Dict[str, Dict[str, Any]] = {}

    def refresh_cache(self) -> None:
        """
        Перезагрузить все семена из директории в кэш.
        
        Читает все JSON-файлы из registry_dir и загружает их в кэш.
        Если файл содержит невалидный JSON, логирует ошибку и пропускает файл.
        
        Note:
            Этот метод не вызывается автоматически в __init__.
            Вызывайте его явно после создания экземпляра класса.
            
        Example:
            >>> registry = SeedRegistry("./seeds")
            >>> registry.refresh_cache()
        """
        self._cache = {}
        for filename in os.listdir(self.registry_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.registry_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        proof_name = filename.replace(".json", "")
                        self._cache[proof_name] = data
                    except json.JSONDecodeError:
                        logger.warning(
                            get_t("error_reading_file", self.lang, filename=filename)
                        )

    def register_proof(
        self,
        proof_name: str,
        model_type: str,
        layer_idx: int,
        master_indices: List[int],
        scale: float,
        model_hidden_size: Optional[int] = None,
        model_name: Optional[str] = None,
        scaled_model_path: Optional[str] = None
    ) -> None:
        """
        Зарегистрировать новое семя (proof) в реестре.
        
        Создаёт JSON-файл с данными семени и добавляет его в кэш.
        
        Args:
            proof_name: Имя семени (используется как имя файла без расширения)
            model_type: Тип архитектуры модели (например, "qwen2", "gpt2")
            layer_idx: Индекс слоя, к которому применено масштабирование
            master_indices: Список индексов мастер-нейронов
            scale: Коэффициент масштабирования весов
            model_hidden_size: Размер скрытого слоя модели (опционально)
            model_name: Имя или путь к модели (опционально)
            scaled_model_path: Путь к предварительно масштабированной модели (опционально)
            
        Raises:
            OSError: Если не удалось записать файл
            
        Example:
            >>> registry.register_proof(
            ...     proof_name="honesty_seed",
            ...     model_type="qwen2",
            ...     layer_idx=12,
            ...     master_indices=[100, 200, 300],
            ...     scale=1.5,
            ...     model_hidden_size=1024
            ... )
        """
        proof_data = {
            "model_type": model_type,
            "layer_idx": layer_idx,
            "master_indices": master_indices,
            "scale": scale
        }
        if model_hidden_size is not None:
            proof_data["model_hidden_size"] = model_hidden_size
        if model_name is not None:
            proof_data["model_name"] = model_name
        if scaled_model_path is not None:
            proof_data["scaled_model_path"] = scaled_model_path
        
        self._cache[proof_name] = proof_data

        filepath = os.path.join(self.registry_dir, f"{proof_name}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(proof_data, f, indent=2)

        logger.info(
            get_t("proof_registered", self.lang, proof_name=proof_name, filepath=filepath)
        )

    def get_proof(self, proof_name: str) -> Optional[Dict[str, Any]]:
        """
        Получить данные семени по имени.
        
        Args:
            proof_name: Имя семени для поиска
            
        Returns:
            Словарь с данными семени или None, если семя не найдено
            
        Example:
            >>> seed = registry.get_proof("honesty_seed")
            >>> if seed:
            ...     print(seed["scale"])
            1.5
        """
        return self._cache.get(proof_name)

    def list_proofs(self) -> List[str]:
        """
        Получить список всех зарегистрированных семян.
        
        Returns:
            Список имён всех семян в реестре
            
        Example:
            >>> proofs = registry.list_proofs()
            >>> print(proofs)
            ['honesty_seed', 'safety_seed', 'creativity_seed']
        """
        return list(self._cache.keys())

    def is_compatible(self, proof_name: str, current_model_type: str) -> bool:
        """
        Проверить совместимость семени с типом модели.
        
        Простая проверка совпадения model_type семени с типом текущей модели.
        
        Args:
            proof_name: Имя семени для проверки
            current_model_type: Тип текущей модели (например, "qwen2")
            
        Returns:
            True если семя совместимо с моделью, False в противном случае
            
        Example:
            >>> if registry.is_compatible("honesty_seed", "qwen2"):
            ...     print("Семя совместимо с моделью")
        """
        proof = self.get_proof(proof_name)
        if not proof:
            return False
        return proof.get("model_type") == current_model_type
