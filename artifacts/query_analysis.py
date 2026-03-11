# python query_analysis.py --output-dir . --base-query "Registrska tablica kot osebni podatek" --top-k 20


# Če pa želiš eksplicitno prisiliti pravi model, lahko daš:
# python query_analysis.py --output-dir . --model google/embeddinggemma-300m --base-query "Registrska tablica kot osebni podatek" --top-k 20

#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import faiss
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a vector DB with multiple paraphrased queries and visualize similarity on PCA space."
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts_fast",
        help="Directory containing metadata.json, embeddings.npy, ip_opinions.index, manifest.json",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional override for SentenceTransformer model name. "
             "If omitted, model from manifest.json is used.",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("HF_HOME"),
        help="HF cache dir, e.g. $SCRATCH/hf_cache",
    )
    parser.add_argument(
        "--base-query",
        default="Registrska tablica kot osebni podatek",
        help="Base query used for colored PCA visualization",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top results per query",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Optional directory for outputs; defaults to <output-dir>/query_analysis",
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


def load_artifacts(output_dir: str) -> Tuple[List[Dict[str, str]], np.ndarray, faiss.Index, Dict[str, Any]]:
    metadata_path = os.path.join(output_dir, "metadata.json")
    embeddings_path = os.path.join(output_dir, "embeddings.npy")
    index_path = os.path.join(output_dir, "ip_opinions.index")
    manifest_path = os.path.join(output_dir, "manifest.json")

    for path in [metadata_path, embeddings_path, index_path, manifest_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required artifact: {path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    embeddings = np.load(embeddings_path)
    index = faiss.read_index(index_path)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if not isinstance(records, list) or not records:
        raise RuntimeError("metadata.json is empty or invalid")

    if embeddings.ndim != 2 or embeddings.shape[0] != len(records):
        raise RuntimeError(
            f"embeddings.npy shape {embeddings.shape} does not match metadata length {len(records)}"
        )

    if index.ntotal != len(records):
        raise RuntimeError(
            f"FAISS index size {index.ntotal} does not match metadata length {len(records)}"
        )

    return records, embeddings.astype("float32"), index, manifest


def validate_normalized(embeddings: np.ndarray) -> None:
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.all(np.isfinite(norms)):
        raise RuntimeError("Embeddings contain non-finite values")
    if np.min(norms) < 0.95 or np.max(norms) > 1.05:
        raise RuntimeError(
            f"Embeddings are expected to be normalized. min_norm={np.min(norms):.4f}, max_norm={np.max(norms):.4f}"
        )


def default_query_variants(base_query: str) -> List[str]:
    variants = [
        base_query,
        "Ali je registrska tablica osebni podatek",
        "Registrska oznaka vozila kot osebni podatek",
        "Je registrska številka vozila osebni podatek",
        "Avtomobilska registrska tablica in varstvo osebnih podatkov",
        "Identifikacija posameznika prek registrske tablice",
        "Obdelava registrskih tablic vozil kot osebnih podatkov",
        "Ali se podatek o registrski tablici šteje za osebni podatek",
        "Vehicle license plate as personal data",
        "Registration plate personal data GDPR",
    ]
    seen = set()
    unique = []
    for q in variants:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    return unique


def encode_query(model, query: str, expected_dim: int, truncate_dim: Optional[int] = None) -> np.ndarray:
    """
    Encode query in a way that is compatible with the model used at build time.
    Supports both MiniLM-style models and EmbeddingGemma-style query/document APIs.
    """
    if hasattr(model, "encode_query"):
        if truncate_dim is not None:
            vec = model.encode_query(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True,
                truncate_dim=truncate_dim,
            )
        else:
            vec = model.encode_query(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
    else:
        # fallback za modele brez encode_query
        if truncate_dim is not None:
            vec = model.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True,
                truncate_dim=truncate_dim,
            )
        else:
            vec = model.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

    vec = np.asarray(vec, dtype="float32")

    if vec.ndim != 1:
        raise RuntimeError(f"Unexpected query embedding shape: {vec.shape}")

    if vec.shape[0] != expected_dim:
        raise RuntimeError(
            f"Query dim {vec.shape[0]} != embedding dim {expected_dim}. "
            f"Likely cause: using a different model than the one used to build the index."
        )

    return vec


def top_k_results(index: faiss.Index, records: List[Dict[str, str]], qvec: np.ndarray, k: int) -> List[Dict[str, Any]]:
    q2 = qvec.reshape(1, -1)
    scores, indices = index.search(q2, min(k, len(records)))
    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        rec = records[int(idx)]
        results.append(
            {
                "rank": rank,
                "score": float(score),
                "title": rec["title"],
                "link": rec["link"],
                "index": int(idx),
            }
        )
    return results


def reciprocal_rank_fusion(results_per_query: Dict[str, List[Dict[str, Any]]], rrf_k: int = 60) -> List[Dict[str, Any]]:
    scores: Dict[int, float] = {}
    example_payload: Dict[int, Dict[str, Any]] = {}

    for query, results in results_per_query.items():
        for item in results:
            doc_idx = item["index"]
            rank = item["rank"]
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (rrf_k + rank)
            if doc_idx not in example_payload:
                example_payload[doc_idx] = item

    fused = []
    for doc_idx, score in scores.items():
        payload = dict(example_payload[doc_idx])
        payload["rrf_score"] = float(score)
        fused.append(payload)

    fused.sort(key=lambda x: (-x["rrf_score"], x["rank"]))
    for new_rank, item in enumerate(fused, start=1):
        item["consensus_rank"] = new_rank
    return fused


def overlap_at_k(results_a: List[Dict[str, Any]], results_b: List[Dict[str, Any]]) -> Dict[str, float]:
    set_a = {x["index"] for x in results_a}
    set_b = {x["index"] for x in results_b}
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    jaccard = inter / union if union else 0.0
    overlap = inter / min(len(set_a), len(set_b)) if min(len(set_a), len(set_b)) else 0.0
    return {"intersection": inter, "jaccard": jaccard, "overlap_ratio": overlap}


def save_query_results(results_dir: str, query: str, results: List[Dict[str, Any]]) -> str:
    safe_name = "".join(c.lower() if c.isalnum() else "_" for c in query).strip("_")
    safe_name = safe_name[:120] or "query"
    path = os.path.join(results_dir, f"top20_{safe_name}.json")
    payload = {
        "query": query,
        "top_results": [
            {k: v for k, v in item.items() if k != "index"}
            for item in results
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def build_similarity_plot(
    results_dir: str,
    records: List[Dict[str, str]],
    embeddings: np.ndarray,
    qvec: np.ndarray,
    base_query: str,
) -> None:
    from sklearn.decomposition import PCA
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(records) < 2:
        print("[WARN] Skipping PCA similarity plot because fewer than 2 documents are available")
        return

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)
    query_coord = pca.transform(qvec.reshape(1, -1))[0]

    similarities = embeddings @ qvec

    fig = plt.figure(figsize=(15, 11))
    ax = fig.add_subplot(111)

    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=similarities,
        cmap="viridis",
        s=28,
        alpha=0.88,
        edgecolors="none",
    )

    # top_idx = np.argsort(-similarities)[:5]
    # for idx in top_idx:
    #     x, y = coords[idx]
    #     title = records[idx]["title"]
    #     short_title = title if len(title) <= 55 else title[:52] + "..."
    #     ax.annotate(
    #         short_title,
    #         (x, y),
    #         textcoords="offset points",
    #         xytext=(6, 4),
    #         fontsize=8,
    #         alpha=0.95,
    #         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
    #     )

    ax.scatter(
        [query_coord[0]],
        [query_coord[1]],
        marker="*",
        s=450,
        c="red",
        edgecolors="black",
        linewidths=0.8,
        label="Query",
        zorder=5,
    )

    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Cosine similarity to base query", rotation=90)

    ax.set_title(f"PCA prostor dokumentov, obarvan glede na podobnost queryu:\n{base_query}")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% variance)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "pca_similarity_base_query.png"), dpi=200)
    plt.close(fig)

    points = []
    for rec, xy, sim in zip(records, coords, similarities):
        points.append(
            {
                "title": rec["title"],
                "link": rec["link"],
                "x": float(xy[0]),
                "y": float(xy[1]),
                "similarity_to_base_query": float(sim),
            }
        )

    payload = {
        "base_query": base_query,
        "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "query_point": {
            "x": float(query_coord[0]),
            "y": float(query_coord[1]),
        },
        "points": points,
    }

    with open(os.path.join(results_dir, "pca_similarity_base_query.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_stability_report(
    results_dir: str,
    query_variants: List[str],
    results_per_query: Dict[str, List[Dict[str, Any]]],
) -> None:
    matrix = []
    for qa in query_variants:
        row = {"query": qa, "comparisons": []}
        for qb in query_variants:
            stats = overlap_at_k(results_per_query[qa], results_per_query[qb])
            row["comparisons"].append({"other_query": qb, **stats})
        matrix.append(row)

    base_query = query_variants[0]
    base_results = results_per_query[base_query]
    avg_jaccard = 0.0
    count = 0
    for q in query_variants[1:]:
        stats = overlap_at_k(base_results, results_per_query[q])
        avg_jaccard += stats["jaccard"]
        count += 1
    avg_jaccard = avg_jaccard / count if count else 1.0

    report = {
        "base_query": base_query,
        "num_query_variants": len(query_variants),
        "average_jaccard_vs_base_top20": avg_jaccard,
        "interpretation": (
            "Višji overlap med parafrazami pomeni bolj stabilen pravni iskalnik. "
            "To je uporaben keypoint za predstavitev robustnosti na različne formulacije vprašanja."
        ),
        "pairwise_top20_overlap": matrix,
    }

    with open(os.path.join(results_dir, "stability_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def validate_outputs(results_dir: str, query_variants: List[str]) -> None:
    required = [
        os.path.join(results_dir, "pca_similarity_base_query.png"),
        os.path.join(results_dir, "pca_similarity_base_query.json"),
        os.path.join(results_dir, "stability_report.json"),
        os.path.join(results_dir, "consensus_top20_rrf.json"),
    ]
    for p in required:
        if not os.path.exists(p):
            raise RuntimeError(f"Missing output: {p}")

    for query in query_variants:
        safe_name = "".join(c.lower() if c.isalnum() else "_" for c in query).strip("_")
        safe_name = safe_name[:120] or "query"
        path = os.path.join(results_dir, f"top20_{safe_name}.json")
        if not os.path.exists(path):
            raise RuntimeError(f"Missing per-query output: {path}")

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        top_results = payload.get("top_results", [])
        if not top_results:
            raise RuntimeError(f"No results stored in {path}")
        if len(top_results) > 20:
            raise RuntimeError(f"Too many results in {path}: {len(top_results)}")

    print("[OK] Output validation successful")


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir or os.path.join(args.output_dir, "query_analysis")
    ensure_dir(results_dir)

    records, embeddings, index, manifest = load_artifacts(args.output_dir)
    validate_normalized(embeddings)

    expected_dim = int(manifest["embedding_dim"])
    model_name = args.model or manifest.get("model_name")
    if not model_name:
        raise RuntimeError("No model specified and manifest.json does not contain model_name")

    truncate_dim = manifest.get("truncate_dim")
    print(f"[INFO] Using model from manifest/args: {model_name}")
    print(f"[INFO] Expected embedding dim: {expected_dim}")
    print(f"[INFO] Manifest truncate_dim: {truncate_dim}")

    model = load_model(model_name, args.cache_dir)

    query_variants = default_query_variants(args.base_query)
    results_per_query: Dict[str, List[Dict[str, Any]]] = {}
    query_vectors: Dict[str, np.ndarray] = {}

    for query in query_variants:
        qvec = encode_query(model, query, expected_dim, truncate_dim=truncate_dim)
        query_vectors[query] = qvec
        results = top_k_results(index, records, qvec, args.top_k)
        results_per_query[query] = results
        out_path = save_query_results(results_dir, query, results)
        print(f"[INFO] Saved top results for query: {query}")
        print(f"[INFO] -> {out_path}")

    build_similarity_plot(
        results_dir=results_dir,
        records=records,
        embeddings=embeddings,
        qvec=query_vectors[args.base_query],
        base_query=args.base_query,
    )

    fused = reciprocal_rank_fusion(results_per_query)
    fused_path = os.path.join(results_dir, "consensus_top20_rrf.json")
    with open(fused_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "Reciprocal Rank Fusion",
                "purpose": "Consensus ranking across paraphrases of the same legal question",
                "top_results": [
                    {k: v for k, v in item.items() if k != "index"}
                    for item in fused[:20]
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[INFO] Saved consensus ranking -> {fused_path}")

    build_stability_report(results_dir, query_variants, results_per_query)
    validate_outputs(results_dir, query_variants)

    print("\n[RESULTS PREVIEW]\n")
    for query in query_variants:
        print(f"QUERY: {query}")
        for item in results_per_query[query][:5]:
            print(f"  {item['rank']}. {item['score']:.4f} | {item['title']}")
        print()

    print("[OK] Done")


if __name__ == "__main__":
    main()