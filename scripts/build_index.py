import logging
from pathlib import Path
import yaml
from rag.loader import load_chunks
from rag.vectorstore import build_vectorstore


def main() -> None:

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    config_path = Path(__file__).parent.parent / "config.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    try:
        docs = load_chunks(*cfg["paths"])
        vs = build_vectorstore(documents=docs, model_name=cfg["vectorstore"]["model"])
        output_dir = Path(cfg["vectorstore"]["path"])
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(output_dir))
        logger.info(f"Vectorstore сохранен в {output_dir}")
    except Exception as e:
        logger.exception("Ошибка при сборке векторного индекса: %s", e)
        raise


if __name__ == "__main__":
    main()
