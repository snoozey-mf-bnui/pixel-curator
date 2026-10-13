"""Phase 2: Embedding Visualization & Clustering (additive).

CLIP embeddings (reusing the diversity-clustering cache + optional
loader), 2D projection with a graceful UMAP -> t-SNE -> PCA fallback
chain, lightweight clustering (HDBSCAN -> scikit-learn KMeans ->
pure-python KMeans), cluster summaries with professional review
priorities, and a JSON payload the UI scatter viewer and CLI share.
"""

from .clustering import cluster_embeddings, cluster_summaries
from .payload import build_viz_payload, render_viz_summary, viz_cache_dir
from .projection import available_methods, project_2d

__all__ = ["build_viz_payload", "render_viz_summary", "viz_cache_dir",
           "project_2d", "available_methods", "cluster_embeddings",
           "cluster_summaries"]
