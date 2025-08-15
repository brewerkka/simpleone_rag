import json
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

DISTANCE_THRESHOLD = 1.0
LINKAGE = "average"
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def semantic_chunk(
    text: str,
    target_sent_count: int = 5,
    distance_threshold: float = DISTANCE_THRESHOLD,
):
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
            if len(chunk.split()) < 10 or len(chunk.split()) > 150:
                continue
            chunks.append(chunk)

    return chunks


def main():
    SRC_PATH = Path("test/ru_server_side_api.json")
    OUT_PATH = Path("test/server_chunks.json")

    with open(SRC_PATH, "r", encoding="utf-8") as f:
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
        section_iter, total=total_sections, desc="processing sections"
    ):
        doc_id = doc.get("id")
        title = doc.get("title")
        sec_id = section.get("section_id")
        heading = section.get("heading")
        text = section.get("content", {}).get("text", "").strip()
        if not text:
            continue

        chunks = semantic_chunk(text)
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

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"chunks": all_chunks}, f, ensure_ascii=False, indent=2)

    print(f"{len(all_chunks)} chunks and saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
