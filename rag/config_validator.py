"""Валидация конфигурации проекта."""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, List


def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Валидирует конфигурацию и возвращает (успех, список ошибок)."""
    errors = []

    # Проверка обязательных секций
    required_sections = ["paths", "vectorstore", "llm", "retrieval"]
    for section in required_sections:
        if section not in config:
            errors.append(f"Отсутствует обязательная секция: {section}")

    # Проверка paths
    if "paths" in config:
        if not isinstance(config["paths"], list) or len(config["paths"]) == 0:
            errors.append("'paths' должен быть непустым списком")
        else:
            for path in config["paths"]:
                if not isinstance(path, str):
                    errors.append(f"Путь должен быть строкой: {path}")

    # Проверка vectorstore
    if "vectorstore" in config:
        vs = config["vectorstore"]
        if "path" not in vs:
            errors.append("'vectorstore.path' обязателен")
        if "model" not in vs:
            errors.append("'vectorstore.model' обязателен")

    # Проверка llm
    if "llm" in config:
        llm = config["llm"]
        if "model" not in llm:
            errors.append("'llm.model' обязателен")
        if "api_key_env" not in llm:
            errors.append("'llm.api_key_env' обязателен")
        else:
            api_key = os.getenv(llm["api_key_env"])
            if not api_key:
                errors.append(
                    f"Переменная окружения {llm['api_key_env']} не установлена"
                )

    # Проверка retrieval
    if "retrieval" in config:
        if "k" not in config["retrieval"]:
            errors.append("'retrieval.k' обязателен")
        elif not isinstance(config["retrieval"]["k"], int) or config["retrieval"]["k"] < 1:
            errors.append("'retrieval.k' должен быть положительным целым числом")

    return len(errors) == 0, errors


def load_and_validate_config(config_path: Path = None) -> Dict[str, Any]:
    """Загружает и валидирует конфигурацию."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    is_valid, errors = validate_config(config)

    if not is_valid:
        error_msg = "Ошибки валидации конфигурации:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)

    return config

