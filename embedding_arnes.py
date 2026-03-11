#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer

DEFAULT_DATA_DIR = "database"
DEFAULT_OUTPUT_DIR = "artifacts"
DEFAULT_MODEL_NAME = "google/embeddinggemma-300m"


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_name: str, cache_dir: str | None = None) -> SentenceTransformer:
    device = get_device()
    print(f"[INFO] Loading model: {model_name}")
    print(f"[INFO] Device: {device}")
    if cache_dir:
        print(f"[INFO] HF cache dir: {cache_dir}")
    model = SentenceTransformer(model_name, cache_folder=cache_dir, device=device)
    return model


def load_category(category_path: str) -> Tuple[List[str], List[str]]:
    titles: List[str] = []
    links: List[str] = []

    for file_name in sorted(os.listdir(category_path)):
        if not file_name.endswith(".json"):
            continue

        file_path = os.path.join(category_path, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        title = data.get("title")
        link = data.get("link")

        if not title or not link:
            print(f"[WARN] Skipping {file_path} because title or link is missing")
            continue

        titles.append(str(title))
        links.append(str(link))

    return titles, links


def build_document_texts(titles: List[str]) -> List[str]:
    # Ker imate trenutno samo naslove, dokument prompt oblikujemo iz naslova.
    # EmbeddingGemma priporoča document prompt "title: ... | text: ...".
    return [f"title: {title} | text: {title}" for title in titles]


def encode_documents(
    model: SentenceTransformer,
    titles: List[str],
    batch_size: int = 64,
    truncate_dim: int | None = None,
) -> np.ndarray:
    doc_texts = build_document_texts(titles)

    # Če encode_document ni na voljo, pade na encode
    if hasattr(model, "encode_document"):
        vectors = model.encode_document(
            doc_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
            truncate_dim=truncate_dim,
        )
    else:
        vectors = model.encode(
            doc_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
            truncate_dim=truncate_dim,
        )

    return vectors.astype("float32")


def encode_query(
    model: SentenceTransformer,
    query: str,
    truncate_dim: int | None = None,
) -> np.ndarray:
    if hasattr(model, "encode_query"):
        vec = model.encode_query(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            truncate_dim=truncate_dim,
        )
    else:
        # fallback: ročno uporabi priporočeni query prompt
        prompted = f"task: search result | query: {query}"
        vec = model.encode(
            prompted,
            convert_to_numpy=True,
            normalize_embeddings=True,
            truncate_dim=truncate_dim,
        )

    return np.asarray(vec, dtype="float32")


def save_index(index: faiss.Index, output_dir: str, category_name: str) -> str:
    ensure_dir(output_dir)
    index_path = os.path.join(output_dir, f"{category_name}.index")
    faiss.write_index(index, index_path)
    return index_path


def save_metadata(metadata: Dict, output_dir: str) -> str:
    ensure_dir(output_dir)
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return metadata_path


def build_index_for_category(
    model: SentenceTransformer,
    category_name: str,
    titles: List[str],
    output_dir: str,
    batch_size: int,
    truncate_dim: int | None,
) -> faiss.Index:
    if not titles:
        raise ValueError(f"Category '{category_name}' has no valid documents")

    vectors = encode_documents(
        model=model,
        titles=titles,
        batch_size=batch_size,
        truncate_dim=truncate_dim,
    )

    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    index_path = save_index(index, output_dir, category_name)
    print(f"[INFO] {category_name} -> {index.ntotal} documents -> {index_path}")
    return index


def build_all_databases(
    data_dir: str,
    output_dir: str,
    model_name: str,
    batch_size: int,
    truncate_dim: int | None,
    cache_dir: str | None,
) -> None:
    model = load_model(model_name, cache_dir=cache_dir)

    metadata: Dict[str, List[Dict[str, str]]] = {}
    category_count = 0

    for category in sorted(os.listdir(data_dir)):
        category_path = os.path.join(data_dir, category)

        if not os.path.isdir(category_path):
            continue

        titles, links = load_category(category_path)
        if not titles:
            print(f"[WARN] Skipping empty category: {category}")
            continue

        build_index_for_category(
            model=model,
            category_name=category,
            titles=titles,
            output_dir=output_dir,
            batch_size=batch_size,
            truncate_dim=truncate_dim,
        )

        metadata[category] = [
            {"title": t, "link": l}
            for t, l in zip(titles, links)
        ]
        category_count += 1

    metadata_path = save_metadata(metadata, output_dir)
    print(f"[OK] Built {category_count} categories")
    print(f"[OK] Metadata saved to {metadata_path}")


def search(
    query: str,
    category: str,
    output_dir: str,
    model_name: str,
    k: int,
    truncate_dim: int | None,
    cache_dir: str | None,
) -> None:
    model = load_model(model_name, cache_dir=cache_dir)

    index_path = os.path.join(output_dir, f"{category}.index")
    metadata_path = os.path.join(output_dir, "metadata.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index not found: {index_path}")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    index = faiss.read_index(index_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if category not in metadata:
        raise KeyError(f"Category '{category}' not found in metadata")

    qvec = encode_query(model, query, truncate_dim=truncate_dim).reshape(1, -1)
    D, I = index.search(qvec, min(k, index.ntotal))

    print("\n[RESULTS]\n")
    for rank, (score, idx) in enumerate(zip(D[0], I[0]), start=1):
        item = metadata[category][int(idx)]
        print(f"{rank}. score={score:.4f}")
        print(item["title"])
        print(item["link"])
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build/search FAISS indices for title retrieval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build all category indices")
    build_parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    build_parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    build_parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    build_parser.add_argument("--batch-size", type=int, default=64)
    build_parser.add_argument("--truncate-dim", type=int, default=None, choices=[128, 256, 512, 768, None])
    build_parser.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))

    search_parser = subparsers.add_parser("search", help="Search one category")
    search_parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    search_parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    search_parser.add_argument("--category", required=True)
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("-k", type=int, default=5)
    search_parser.add_argument("--truncate-dim", type=int, default=None, choices=[128, 256, 512, 768, None])
    search_parser.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "build":
        build_all_databases(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            model_name=args.model,
            batch_size=args.batch_size,
            truncate_dim=args.truncate_dim,
            cache_dir=args.cache_dir,
        )
    elif args.command == "search":
        search(
            query=args.query,
            category=args.category,
            output_dir=args.output_dir,
            model_name=args.model,
            k=args.k,
            truncate_dim=args.truncate_dim,
            cache_dir=args.cache_dir,
        )
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()