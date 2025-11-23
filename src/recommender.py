# All comments in English.
from typing import List, Tuple, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def is_actually_pants(item_meta: Dict) -> bool:
    """( Tee)"""
    text = (item_meta.get('title', '') + ' ' +
            item_meta.get('description', '')).lower()
    pants_keywords = ['pants', 'jeans', 'trousers',
                      'shorts', 'leggings', 'joggers']
    # ,
    has_pants = any(kw in text for kw in pants_keywords)
    has_top = any(kw in text for kw in [
                  'top', 'shirt', 'blouse', 'tee', 'jacket', 'sweater', 'cardigan'])
    #
    return has_pants and not has_top


def recommend_topk(query_vec: np.ndarray, gallery_matrix: np.ndarray, meta: List[Dict], k: int = 5,
                   query_category: str = None, complementary_mode: bool = True, query_gender: str = None) -> List[Tuple[Dict, float]]:
    """
    Recommend fashion items based on cosine similarity with gender filtering.

    Args:
        query_vec: Feature vector of query image
        gallery_matrix: Feature matrix of gallery items
        meta: Metadata list for gallery items
        k: Number of recommendations
        query_category: Category of query item (e.g., 'Blouse', 'Skirt')
        complementary_mode: If True, recommend complementary items for outfit matching
        query_gender: Gender of query item ('Male', 'Female', 'Unisex') for filtering

    Returns:
        List of (metadata, similarity) tuples
    """
    # Define complementary categories for outfit matching
    TOPS = ['Blazer', 'Blouse', 'Tee']
    BOTTOMS = ['Skirt', 'Pants', 'Shorts']  #
    DRESSES = ['Dress']  # Dresses are standalone outfits

    sims = cosine_similarity(query_vec.reshape(1, -1), gallery_matrix)[0]

    if complementary_mode and query_category:
        # Outfit matching mode: recommend complementary items
        if query_category in TOPS:
            # Query is a top -> recommend bottoms (pants/shorts for male, all bottoms for female)
            target_categories = BOTTOMS
            if query_gender == 'Male':
                # For male tops, recommend pants and shorts (no skirts)
                target_keywords = ['pants', 'jeans',
                                   'trouser', 'shorts', 'joggers', 'leggings']
            else:
                # For female/unisex tops, recommend all bottoms including skirts
                target_keywords = ['skirt', 'bottom', 'pants',
                                   'jeans', 'trouser', 'shorts', 'leggings']
        elif query_category in BOTTOMS:
            # Query is a bottom -> recommend tops
            target_categories = TOPS
            target_keywords = ['blazer', 'blouse', 'top',
                               'shirt', 'tee', 'jacket', 'sweater']
        elif query_category in DRESSES:
            # Dress is complete outfit -> recommend similar dresses
            target_categories = DRESSES
            target_keywords = ['dress', 'gown', 'frock']
        else:
            # Unknown category -> use similarity only
            target_categories = None
            target_keywords = []

        if target_categories:
            # Filter gallery items by complementary categories AND gender
            filtered_indices = []
            for i, item_meta in enumerate(meta):
                # Check if item has category info
                item_category = item_meta.get('category', '')
                item_gender = item_meta.get('gender', 'Unisex')

                # Get path for keyword matching
                path = item_meta.get('image_path', '').lower()

                # : ,
                if target_categories == TOPS and is_actually_pants(item_meta):
                    continue

                # : ,Tee
                is_pants = is_actually_pants(item_meta)
                if target_categories == BOTTOMS and is_pants:
                    # ,
                    exact_match = True
                else:
                    # Check if item matches target categories (exact match)
                    exact_match = any(cat.lower() in item_category.lower()
                                      for cat in target_categories)

                # Check if path contains target keywords (fuzzy match for unlabeled data)
                keyword_match = any(
                    keyword in path for keyword in target_keywords)

                # Gender filtering: only show items matching query gender or Unisex items
                gender_match = True
                if query_gender:
                    if query_gender == 'Male':
                        # Male query: show Male and Unisex items only
                        gender_match = item_gender in ['Male', 'Unisex']
                    elif query_gender == 'Female':
                        # Female query: show Female and Unisex items only
                        gender_match = item_gender in ['Female', 'Unisex']
                    else:
                        # Unisex query: show all genders
                        gender_match = True

                if (exact_match or keyword_match) and gender_match:
                    filtered_indices.append(i)

            if len(filtered_indices) > 0:
                # Sort filtered items by similarity
                filtered_sims = [(idx, sims[idx]) for idx in filtered_indices]
                filtered_sims.sort(key=lambda x: x[1], reverse=True)
                topk_idx = [idx for idx, _ in filtered_sims[:k]]
            else:
                # Fallback: no complementary items found, but STILL apply gender filter
                # Try to filter out same category items while respecting gender
                same_keywords = []
                if query_category in TOPS:
                    same_keywords = ['blazer', 'blouse', 'top',
                                     'shirt', 'tee', 'jacket', 'sweater']
                elif query_category in BOTTOMS:
                    same_keywords = ['skirt', 'bottom',
                                     'pants', 'jeans', 'trouser']
                elif query_category in DRESSES:
                    same_keywords = []  # Keep dresses for dresses

                # Filter out same category items AND apply gender filter
                different_indices = []
                for i, item_meta in enumerate(meta):
                    path = item_meta.get('image_path', '').lower()
                    item_category = item_meta.get('category', '')
                    is_same = any(keyword in path for keyword in same_keywords)

                    #
                    if query_category in TOPS and item_category in TOPS:
                        is_same = True
                    elif query_category in BOTTOMS and item_category in BOTTOMS:
                        is_same = True

                    # Apply gender filter
                    item_gender = item_meta.get('gender', 'Unisex')
                    gender_match = True
                    if query_gender:
                        if query_gender == 'Male':
                            gender_match = item_gender in ['Male', 'Unisex']
                        elif query_gender == 'Female':
                            gender_match = item_gender in ['Female', 'Unisex']
                        else:
                            gender_match = True

                    # Include if: (not same category OR no keywords) AND gender matches
                    if (not is_same or not same_keywords) and gender_match:
                        different_indices.append(i)

                if len(different_indices) > 0:
                    different_sims = [(idx, sims[idx])
                                      for idx in different_indices]
                    different_sims.sort(key=lambda x: x[1], reverse=True)
                    topk_idx = [idx for idx, _ in different_sims[:k]]
                else:
                    # Last resort: apply gender filter only (ignore category)
                    if query_gender:
                        gender_only_indices = []
                        for i, item_meta in enumerate(meta):
                            item_gender = item_meta.get('gender', 'Unisex')
                            if query_gender == 'Male' and item_gender in ['Male', 'Unisex']:
                                gender_only_indices.append(i)
                            elif query_gender == 'Female' and item_gender in ['Female', 'Unisex']:
                                gender_only_indices.append(i)
                            elif query_gender not in ['Male', 'Female']:
                                gender_only_indices.append(i)

                        if len(gender_only_indices) > 0:
                            gender_sims = [(idx, sims[idx])
                                           for idx in gender_only_indices]
                            gender_sims.sort(key=lambda x: x[1], reverse=True)
                            topk_idx = [idx for idx, _ in gender_sims[:k]]
                        else:
                            # Absolutely last resort: use top similarity (no filters)
                            topk_idx = np.argsort(-sims)[:k]
                    else:
                        topk_idx = np.argsort(-sims)[:k]
        else:
            topk_idx = np.argsort(-sims)[:k]
    else:
        # Regular similarity mode - but still apply gender filter
        if query_gender:
            gender_filtered_indices = []
            for i, item_meta in enumerate(meta):
                item_gender = item_meta.get('gender', 'Unisex')
                if query_gender == 'Male' and item_gender in ['Male', 'Unisex']:
                    gender_filtered_indices.append(i)
                elif query_gender == 'Female' and item_gender in ['Female', 'Unisex']:
                    gender_filtered_indices.append(i)
                elif query_gender not in ['Male', 'Female']:
                    gender_filtered_indices.append(i)

            if len(gender_filtered_indices) > 0:
                gender_sims = [(idx, sims[idx])
                               for idx in gender_filtered_indices]
                gender_sims.sort(key=lambda x: x[1], reverse=True)
                topk_idx = [idx for idx, _ in gender_sims[:k]]
            else:
                topk_idx = np.argsort(-sims)[:k]
        else:
            topk_idx = np.argsort(-sims)[:k]

    return [(meta[i], float(sims[i])) for i in topk_idx]
