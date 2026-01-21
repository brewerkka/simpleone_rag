import logging
from pathlib import Path
from rag.loader import load_chunks
from rag.vectorstore import build_vectorstore
from rag.config_validator import load_and_validate_config


def main() -> None:
    """Строит векторный индекс из чанков."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        cfg = load_and_validate_config()
        
        logger.info("Загрузка чанков...")
        docs = load_chunks(*cfg["paths"])
        logger.info(f"Загружено {len(docs)} документов")
        
        logger.info("Построение векторного индекса...")
        vs = build_vectorstore(documents=docs, model_name=cfg["vectorstore"]["model"])
        
        output_dir = Path(cfg["vectorstore"]["path"])
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(output_dir))
        logger.info(f"Векторный индекс сохранен в {output_dir}")
    except Exception as e:
        logger.exception("Ошибка при сборке векторного индекса: %s", e)
        raise


if __name__ == "__main__":
    main()
