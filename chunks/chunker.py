import json
import logging
import yaml
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import nltk
from nltk.tokenize import sent_tokenize
import warnings
from tqdm import tqdm
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
nltk.download("punkt_tab", quiet=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LINKAGE = "average"
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def semantic_chunk(
    text: str,
    target_sent_count: int = 5,
    distance_threshold: float = 1.0,
    min_words: int = 10,
    max_words: int = 150,
):
    """Разбивает текст на семантические чанки."""
    sents = sent_tokenize(text)
    if len(sents) <= target_sent_count:
        return [" ".join(sents)]

    embeddings = embedder.encode(sents, convert_to_numpy=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normed = embeddings / np.clip(norms, a_min=1e-8, a_max=None)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage=LINKAGE,
        metric="euclidean",
    )
    labels = clustering.fit_predict(embeddings_normed)

    chunks = []
    for cluster_id in sorted(set(labels)):
        inds = [i for i, lab in enumerate(labels) if lab == cluster_id]
        cluster_sents = [sents[i] for i in sorted(inds)]

        for i in range(0, len(cluster_sents), target_sent_count):
            chunk = " ".join(cluster_sents[i : i + target_sent_count])
            word_count = len(chunk.split())
            if word_count < min_words or word_count > max_words:
                continue
            chunks.append(chunk)

    return chunks


def process_file(
    src_path: Path,
    out_path: Path,
    chunking_config: dict,
):
    """Обрабатывает один JSON файл и создает чанки."""
    if not src_path.exists():
        raise FileNotFoundError(f"Исходный файл не найден: {src_path}")

    logger.info(f"Обработка файла: {src_path}")
    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_chunks = []
    total_sections = sum(
        len(doc.get("sections", [])) for doc in data.get("documents", [])
    )

    section_iter = (
        (doc, section)
        for doc in data.get("documents", [])
        for section in doc.get("sections", [])
    )
    for doc, section in tqdm(
        section_iter, total=total_sections, desc=f"Обработка {src_path.name}"
    ):
        doc_id = doc.get("id")
        title = doc.get("title")
        sec_id = section.get("section_id")
        heading = section.get("heading")
        text = section.get("content", {}).get("text", "").strip()
        if not text:
            continue

        chunks = semantic_chunk(
            text,
            target_sent_count=chunking_config.get("target_sent_count", 5),
            distance_threshold=chunking_config.get("distance_threshold", 1.0),
            min_words=chunking_config.get("min_words", 10),
            max_words=chunking_config.get("max_words", 150),
        )
        for idx, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "text": chunk,
                    "metadata": {
                        "doc_id": doc_id,
                        "title": title,
                        "section_id": sec_id,
                        "heading": heading,
                        "chunk_index": idx,
                        "source": f"{doc_id}/{sec_id}",
                    },
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"chunks": all_chunks}, f, ensure_ascii=False, indent=2)

    logger.info(f"Создано {len(all_chunks)} чанков, сохранено в {out_path}")
    return len(all_chunks)


def main():
    """Главная функция для обработки всех файлов из конфигурации."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_data = cfg.get("source_data", {})
    chunking_config = cfg.get("chunking", {})

    # Маппинг исходных файлов на выходные
    file_mapping = {
        "client_side_api": "chunks/client_chunks.json",
        "server_side_api": "chunks/server_chunks.json",
        "widgets_side_api": "chunks/widgets_chunks.json",
    }

    total_chunks = 0
    for key, output_file in file_mapping.items():
        if key not in source_data:
            logger.warning(f"Пропущен ключ {key} в конфигурации")
            continue

        src_path = Path(source_data[key])
        out_path = Path(output_file)

        try:
            chunks_count = process_file(src_path, out_path, chunking_config)
            total_chunks += chunks_count
        except Exception as e:
            logger.error(f"Ошибка при обработке {key}: {e}")
            raise

    logger.info(f"Всего создано {total_chunks} чанков из {len(file_mapping)} файлов")


if __name__ == "__main__":
    main()
