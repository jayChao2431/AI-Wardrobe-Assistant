# All comments in English.
from typing import List, Tuple, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def recommend_topk(query_vec: np.ndarray, gallery_matrix: np.ndarray, meta: List[Dict], k: int = 5) -> List[Tuple[Dict, float]]:
    sims = cosine_similarity(query_vec.reshape(1, -1), gallery_matrix)[0]
    topk_idx = np.argsort(-sims)[:k]
    return [(meta[i], float(sims[i])) for i in topk_idx]
