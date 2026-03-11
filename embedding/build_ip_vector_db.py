# BUILD
# python build_ip_vector_db.py build \
#   --input-json ip_mnenja.json \
#   --output-dir artifacts_test \
#   --batch-size 8 \
#   --truncate-dim 128

# SEARCH
# python build_ip_vector_db.py search \
#   --output-dir artifacts_test \
#   --query "mladoletnik in družbeno omrežje" \
#   -k 2 \
#   --truncate-dim 128


#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build/search a FAISS vector DB for IP opinions"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build",
        help="Build embeddings, FAISS index, metadata and PCA visualization",
    )
    build.add_argument(
        "--input-json",
        required=True,
        help="Path to input JSON file with a list of {title, link}",
    )
    build.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory for index and outputs",
    )
    build.add_argument(
        "--model",
        default="google/embeddinggemma-300m",
        help="SentenceTransformer model name",
    )
    build.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size",
    )
    build.add_argument(
        "--truncate-dim",
        type=int,
        default=256,
        choices=[128, 256, 512, 768],
        help="Optional truncated embedding dimension",
    )
    build.add_argument(
        "--cache-dir",
        default=os.environ.get("HF_HOME"),
        help="HF cache dir, e.g. $SCRATCH/hf_cache",
    )
    build.add_argument(
        "--plot-title",
        default="IP mnenja ZVOP-2: PCA projekcija embeddingov",
    )

    search = subparsers.add_parser("search", help="Search the built vector DB")
    search.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where index and metadata were saved",
    )
    search.add_argument(
        "--model",
        default="google/embeddinggemma-300m",
        help="SentenceTransformer model name",
    )
    search.add_argument(
        "--query",
        required=True,
        help="Search query",
    )
    search.add_argument(
        "-k",
        type=int,
        default=5,
        help="Top-k results",
    )
    search.add_argument(
        "--truncate-dim",
        type=int,
        default=256,
        choices=[128, 256, 512, 768],
        help="Must match build-time dimension",
    )
    search.add_argument(
        "--cache-dir",
        default=os.environ.get("HF_HOME"),
        help="HF cache dir, e.g. $SCRATCH/hf_cache",
    )

    return parser.parse_args()


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_name: str, cache_dir: Optional[str]):
    from sentence_transformers import SentenceTransformer

    device = get_device()
    print(f"[INFO] Loading model: {model_name}")
    print(f"[INFO] Device: {device}")
    if cache_dir:
        print(f"[INFO] Cache dir: {cache_dir}")

    model = SentenceTransformer(
        model_name,
        cache_folder=cache_dir,
        device=device,
    )
    return model


def load_input_data(input_json: str) -> List[Dict[str, str]]:
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must contain a list of objects")

    cleaned: List[Dict[str, str]] = []
    seen = set()

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"[WARN] Skipping item #{idx}: not an object")
            continue

        title = str(item.get("title", "")).strip()
        link = str(item.get("link", "")).strip()

        if not title or not link:
            print(f"[WARN] Skipping item #{idx}: missing title or link")
            continue

        if link in seen:
            continue

        seen.add(link)
        cleaned.append({"title": title, "link": link})

    if not cleaned:
        raise ValueError("No valid documents found in input JSON")

    print(f"[INFO] Loaded {len(cleaned)} unique documents")
    return cleaned


def build_document_texts(records: List[Dict[str, str]]) -> List[str]:
    # EmbeddingGemma priporoča document-style prompt.
    return [f"title: {rec['title']} | text: {rec['title']}" for rec in records]


def encode_documents(model, records: List[Dict[str, str]], batch_size: int, truncate_dim: int) -> np.ndarray:
    texts = build_document_texts(records)

    if hasattr(model, "encode_document"):
        embs = model.encode_document(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
            truncate_dim=truncate_dim,
        )
    else:
        embs = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
            truncate_dim=truncate_dim,
        )

    embs = np.asarray(embs, dtype="float32")

    if embs.ndim != 2 or embs.shape[0] != len(records):
        raise RuntimeError(f"Unexpected embedding shape: {embs.shape}")

    return embs


def encode_query(model, query: str, truncate_dim: int) -> np.ndarray:
    if hasattr(model, "encode_query"):
        vec = model.encode_query(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            truncate_dim=truncate_dim,
        )
    else:
        prompted = f"task: search result | query: {query}"
        vec = model.encode(
            prompted,
            convert_to_numpy=True,
            normalize_embeddings=True,
            truncate_dim=truncate_dim,
        )

    vec = np.asarray(vec, dtype="float32")

    if vec.ndim != 1:
        raise RuntimeError(f"Unexpected query embedding shape: {vec.shape}")

    return vec


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    if vectors.dtype != np.float32:
        vectors = vectors.astype("float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def save_outputs(
    output_dir: str,
    records: List[Dict[str, str]],
    embeddings: np.ndarray,
    index: faiss.Index,
    truncate_dim: int,
    model_name: str,
) -> None:
    ensure_dir(output_dir)

    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
    faiss.write_index(index, os.path.join(output_dir, "ip_opinions.index"))

    manifest = {
        "model_name": model_name,
        "truncate_dim": truncate_dim,
        "num_documents": len(records),
        "embedding_dim": int(embeddings.shape[1]),
        "normalized": True,
        "faiss_index": "IndexFlatIP",
    }

    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def build_pca_artifacts(output_dir: str, records: List[Dict[str, str]], embeddings: np.ndarray, plot_title: str) -> None:
    from sklearn.decomposition import PCA
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if embeddings.shape[0] < 2:
        print("[WARN] Skipping PCA because fewer than 2 documents are available")
        return

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)

    pca_records: List[Dict[str, Any]] = []
    for rec, xy in zip(records, coords):
        pca_records.append(
            {
                "title": rec["title"],
                "link": rec["link"],
                "x": float(xy[0]),
                "y": float(xy[1]),
            }
        )

    with open(os.path.join(output_dir, "pca_coordinates.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
                "points": pca_records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)
    ax.scatter(coords[:, 0], coords[:, 1], s=14, alpha=0.7)
    ax.set_title(plot_title)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% variance)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "pca_plot.png"), dpi=180)
    plt.close(fig)


def validate_build(output_dir: str, expected_count: int, expected_dim: int) -> None:
    metadata_path = os.path.join(output_dir, "metadata.json")
    embeddings_path = os.path.join(output_dir, "embeddings.npy")
    index_path = os.path.join(output_dir, "ip_opinions.index")
    manifest_path = os.path.join(output_dir, "manifest.json")
    pca_json_path = os.path.join(output_dir, "pca_coordinates.json")
    pca_plot_path = os.path.join(output_dir, "pca_plot.png")

    for p in [metadata_path, embeddings_path, index_path, manifest_path]:
        if not os.path.exists(p):
            raise RuntimeError(f"Missing output artifact: {p}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if len(records) != expected_count:
        raise RuntimeError(
            f"metadata.json contains {len(records)} records, expected {expected_count}"
        )

    embeddings = np.load(embeddings_path)
    if embeddings.shape != (expected_count, expected_dim):
        raise RuntimeError(
            f"embeddings.npy shape {embeddings.shape}, expected {(expected_count, expected_dim)}"
        )

    index = faiss.read_index(index_path)
    if index.ntotal != expected_count:
        raise RuntimeError(
            f"FAISS index contains {index.ntotal} vectors, expected {expected_count}"
        )

    norms = np.linalg.norm(embeddings, axis=1)
    if not np.all(np.isfinite(norms)):
        raise RuntimeError("Non-finite embedding norms found")

    if np.min(norms) < 0.95 or np.max(norms) > 1.05:
        raise RuntimeError(
            f"Embeddings do not appear normalized; "
            f"min_norm={np.min(norms):.4f}, max_norm={np.max(norms):.4f}"
        )

    if expected_count >= 2:
        for p in [pca_json_path, pca_plot_path]:
            if not os.path.exists(p):
                raise RuntimeError(f"Missing PCA artifact: {p}")

    print("[OK] Validation successful")


def run_build(args: argparse.Namespace) -> None:
    records = load_input_data(args.input_json)
    model = load_model(args.model, args.cache_dir)
    embeddings = encode_documents(
        model,
        records,
        batch_size=args.batch_size,
        truncate_dim=args.truncate_dim,
    )
    index = build_faiss_index(embeddings)
    save_outputs(
        args.output_dir,
        records,
        embeddings,
        index,
        args.truncate_dim,
        args.model,
    )
    build_pca_artifacts(args.output_dir, records, embeddings, args.plot_title)
    validate_build(
        args.output_dir,
        expected_count=len(records),
        expected_dim=int(embeddings.shape[1]),
    )

    print(f"[OK] Built vector DB with {len(records)} documents")
    print(f"[OK] Output directory: {args.output_dir}")


def run_search(args: argparse.Namespace) -> None:
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    index_path = os.path.join(args.output_dir, "ip_opinions.index")
    manifest_path = os.path.join(args.output_dir, "manifest.json")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Missing {metadata_path}")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Missing {index_path}")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Missing {manifest_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    index = faiss.read_index(index_path)

    if index.ntotal != len(records):
        raise RuntimeError("Index size does not match metadata size")

    build_dim = int(manifest["embedding_dim"])
    truncate_dim = int(args.truncate_dim)

    if truncate_dim != build_dim:
        raise RuntimeError(
            f"Search truncate-dim ({truncate_dim}) must match "
            f"build embedding_dim ({build_dim})"
        )

    model = load_model(args.model, args.cache_dir)
    qvec = encode_query(model, args.query, truncate_dim=args.truncate_dim).reshape(1, -1)

    if qvec.shape[1] != build_dim:
        raise RuntimeError(f"Query dim {qvec.shape[1]} != index dim {build_dim}")

    k = min(args.k, len(records))
    scores, indices = index.search(qvec, k)

    print("\n[RESULTS]\n")
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        rec = records[int(idx)]
        print(f"{rank}. score={float(score):.4f}")
        print(rec["title"])
        print(rec["link"])
        print()


def main() -> None:
    args = parse_args()

    if args.command == "build":
        run_build(args)
    elif args.command == "search":
        run_search(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()