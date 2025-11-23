"""
 (Ensemble Classifier) - Phase 1
: CLIP +  + 

: +5-8% 
: Pants/Shorts 
"""

import torch
import numpy as np
from PIL import Image
import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class KeywordClassifier:
    """"""

    def __init__(self):
        # 
        self.strong_keywords = {
            'pants': ['pants', 'jeans', 'trousers', 'chinos', 'slacks', 'leggings'],
            'shorts': ['shorts', 'bermuda'],
            'skirt': ['skirt', 'mini-skirt', 'midi-skirt', 'maxi-skirt'],
            'tee': ['t-shirt', 'tee', 'tank', 'cami', 'top'],
            'blouse': ['blouse', 'shirt', 'tunic'],
            'dress': ['dress', 'gown', 'frock', 'maxi-dress', 'midi-dress'],
            'blazer': ['blazer', 'jacket', 'coat', 'cardigan']
        }

        self.weak_keywords = {
            'pants': ['denim', 'cotton', 'casual', 'formal', 'straight', 'skinny'],
            'shorts': ['summer', 'casual', 'beach'],
            'skirt': ['pleated', 'flared', 'pencil'],
            'tee': ['casual', 'cotton', 'graphic', 'basic'],
            'blouse': ['formal', 'office', 'button'],
            'dress': ['evening', 'cocktail', 'party', 'summer'],
            'blazer': ['formal', 'office', 'business', 'professional']
        }

        #  (,)
        self.exclusion_keywords = {
            'pants': ['dress', 'skirt', 'short', 'sleeve'],
            'shorts': ['pants', 'long', 'dress', 'skirt'],
            'skirt': ['pants', 'shorts', 'dress'],
            'tee': ['pants', 'skirt', 'dress', 'shorts'],
            'blouse': ['pants', 'skirt', 'shorts'],
            'dress': ['pants', 'shorts', 'skirt', 'separate'],
            'blazer': ['pants', 'skirt', 'shorts', 'dress']
        }

    def classify(self, text: str, categories: List[str]) -> Dict[str, float]:
        """
        

        Returns:
            Dict[category] -> score (0.0 - 1.0)
        """
        text_lower = text.lower()
        scores = {}

        for category in categories:
            score = 0.0

            #  ( +0.4)
            for keyword in self.strong_keywords.get(category, []):
                if keyword in text_lower:
                    score += 0.4

            #  ( +0.1)
            for keyword in self.weak_keywords.get(category, []):
                if keyword in text_lower:
                    score += 0.1

            #  ( -0.3)
            for keyword in self.exclusion_keywords.get(category, []):
                if keyword in text_lower:
                    score -= 0.3

            #  [0, 1]
            scores[category] = max(0.0, min(1.0, score))

        # Softmax 
        if sum(scores.values()) > 0:
            total = sum(scores.values())
            scores = {k: v/total for k, v in scores.items()}
        else:
            # ,
            uniform = 1.0 / len(categories)
            scores = {k: uniform for k in categories}

        return scores


class PathAnalyzer:
    """"""

    def __init__(self):
        # 
        self.path_patterns = {
            'pants': [r'pant', r'jean', r'trouser', r'bottom', r'lower'],
            'shorts': [r'short'],
            'skirt': [r'skirt'],
            # shirt but not in dress context
            'tee': [r'tee', r'top', r'shirt(?!.*dress)'],
            'blouse': [r'blouse', r'tunic'],
            'dress': [r'dress', r'gown'],
            'blazer': [r'blazer', r'jacket', r'coat', r'outerwear']
        }

    def classify(self, image_path: str, categories: List[str]) -> Dict[str, float]:
        """
        

        Returns:
            Dict[category] -> confidence (0.0 - 1.0)
        """
        path_lower = image_path.lower()
        scores = {}

        for category in categories:
            score = 0.0
            patterns = self.path_patterns.get(category, [])

            for pattern in patterns:
                if re.search(pattern, path_lower):
                    score = 0.8  # 
                    break

            scores[category] = score

        # ,
        if sum(scores.values()) > 0:
            total = sum(scores.values())
            scores = {k: v/total for k, v in scores.items()}
        else:
            # ,
            scores = {k: 0.0 for k in categories}

        return scores


class EnsembleClassifier:
    """
     -  CLIP +  + 

    
    """

    def __init__(self, clip_model, clip_preprocess, categories: List[str], device='cpu'):
        """
        Args:
            clip_model: CLIP 
            clip_preprocess: CLIP 
            categories: 
            device: 
        """
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.categories = categories
        self.device = device

        # 
        self.keyword_classifier = KeywordClassifier()
        self.path_analyzer = PathAnalyzer()

        #  ( A :  CLIP)
        self.weights = {
            'clip': 0.95,      # CLIP  95% (,)
            'keyword': 0.03,   #  3% ()
            'path': 0.02       #  2% ()
        }

        #  CLIP  ( A:  - )
        self.text_prompts = {
            'blazer': "a professional blazer jacket or formal coat with structured shoulders, lapels and buttons, worn as outerwear",
            'blouse': "a women's blouse or dress shirt with collar and buttons, formal top covering torso only",
            'dress': "a one-piece dress or gown covering both torso and legs in single garment, worn from shoulders to knees or ankles",
            'skirt': "a skirt worn around waist covering only lower body from hips to knees or ankles, separate from top",
            'tee': "a casual t-shirt or short-sleeved top with crew or v-neck, covering torso only without collar or buttons",
            'pants': "full length pants, jeans or trousers covering entire legs from waist down to ankles, with two separate leg openings",
            'shorts': "short pants or shorts covering upper legs only, ending above knees, with two separate short leg openings"
        }

        #  CLIP 
        self._precompute_text_features()

    def _precompute_text_features(self):
        """ CLIP """
        import clip

        texts = [self.text_prompts.get(cat, f"a photo of a {cat}")
                 for cat in self.categories]
        text_tokens = clip.tokenize(texts).to(self.device)

        with torch.no_grad():
            self.text_features = self.clip_model.encode_text(text_tokens)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def _get_clip_scores(self, image: Image.Image) -> Dict[str, float]:
        """ CLIP """
        # 
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # 
            image_features = self.clip_model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # 
            similarity = (100.0 * image_features @
                          self.text_features.T).softmax(dim=-1)
            scores = similarity[0].cpu().numpy()

        return {cat: float(score) for cat, score in zip(self.categories, scores)}

    def classify(
        self,
        image: Image.Image,
        text_info: str = "",
        image_path: str = "",
        return_details: bool = False
    ) -> Tuple[str, float, Optional[Dict]]:
        """
         (:  metadata , CLIP)

        Args:
            image: PIL 
            text_info: /
            image_path: 
            return_details: 

        Returns:
            (predicted_category, confidence, details)
            details 
        """
        # 1. CLIP 
        clip_scores = self._get_clip_scores(image)

        #  metadata 
        has_text = text_info and len(text_info.strip()) > 10
        has_path_signal = False

        # 2. 
        if has_text:
            keyword_scores = self.keyword_classifier.classify(
                text_info, self.categories)
            # 
            if max(keyword_scores.values()) > 0.15:  # 
                has_text = True
            else:
                has_text = False  # ,
        else:
            keyword_scores = {cat: 1.0/len(self.categories)
                              for cat in self.categories}

        # 3. 
        if image_path:
            path_scores = self.path_analyzer.classify(
                image_path, self.categories)
            if sum(path_scores.values()) > 0:
                has_path_signal = True
        else:
            path_scores = {cat: 0.0 for cat in self.categories}

        # 4. 
        final_scores = {}

        # , CLIP
        if not has_text and not has_path_signal:
            final_scores = clip_scores.copy()
        else:
            # 
            for cat in self.categories:
                clip_score = clip_scores.get(cat, 0.0)
                keyword_score = keyword_scores.get(cat, 0.0)
                path_score = path_scores.get(cat, 0.0)

                # 
                if has_text and has_path_signal:
                    # 
                    final_scores[cat] = (
                        clip_score * self.weights['clip'] +
                        keyword_score * self.weights['keyword'] +
                        path_score * self.weights['path']
                    )
                elif has_text:
                    # 
                    final_scores[cat] = (
                        clip_score * 0.90 +
                        keyword_score * 0.10
                    )
                elif has_path_signal:
                    # 
                    final_scores[cat] = (
                        clip_score * 0.92 +
                        path_score * 0.08
                    )

        # 5. 
        predicted_category = max(final_scores, key=final_scores.get)
        confidence = final_scores[predicted_category]

        # 6. 
        details = None
        if return_details:
            details = {
                'clip_scores': clip_scores,
                'keyword_scores': keyword_scores,
                'path_scores': path_scores,
                'final_scores': final_scores,
                'weights': self.weights,
                'top3': sorted(final_scores.items(), key=lambda x: -x[1])[:3]
            }

        return predicted_category, confidence, details

    def classify_batch(
        self,
        images: List[Image.Image],
        text_infos: List[str] = None,
        image_paths: List[str] = None
    ) -> List[Tuple[str, float]]:
        """"""
        if text_infos is None:
            text_infos = [""] * len(images)
        if image_paths is None:
            image_paths = [""] * len(images)

        results = []
        for img, text, path in zip(images, text_infos, image_paths):
            category, confidence, _ = self.classify(
                img, text, path, return_details=False)
            results.append((category, confidence))

        return results

    def adjust_weights(self, clip_weight: float, keyword_weight: float, path_weight: float):
        """"""
        total = clip_weight + keyword_weight + path_weight
        self.weights = {
            'clip': clip_weight / total,
            'keyword': keyword_weight / total,
            'path': path_weight / total
        }
        print(f"Weights adjusted: CLIP={self.weights['clip']:.2f}, "
              f"Keyword={self.weights['keyword']:.2f}, Path={self.weights['path']:.2f}")


def test_ensemble_classifier():
    """"""
    import clip

    print("Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    categories = ['blazer', 'blouse', 'dress',
                  'skirt', 'tee', 'pants', 'shorts']

    print("Initializing Ensemble Classifier...")
    ensemble = EnsembleClassifier(model, preprocess, categories, device)

    print("\n" + "="*80)
    print("Ensemble Classifier Ready!")
    print("="*80)
    print(f"Categories: {categories}")
    print(f"Fusion Weights: {ensemble.weights}")
    print("\nReady to classify images with enhanced accuracy! ")

    return ensemble


if __name__ == "__main__":
    # 
    ensemble = test_ensemble_classifier()

    print("\n" + "="*80)
    print("Example Usage:")
    print("="*80)
    print("""
from PIL import Image
from ensemble_classifier import EnsembleClassifier

# 
ensemble = EnsembleClassifier(clip_model, clip_preprocess, categories)

# 
image = Image.open('path/to/image.jpg')
category, confidence, details = ensemble.classify(
    image=image,
    text_info="Blue denim jeans",
    image_path="datasets/images/pants_001.jpg",
    return_details=True
)

print(f"Predicted: {category} (confidence: {confidence:.3f})")
print(f"Top 3: {details['top3']}")
    """)
