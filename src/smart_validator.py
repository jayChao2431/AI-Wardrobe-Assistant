"""
 (Smart Post-processing Validator) - Phase 2


:  30-40% 
:  +  + 
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter
import re


class SmartValidator:
    """ - """

    def __init__(self, categories: List[str]):
        """
        Args:
            categories: 
        """
        self.categories = categories

        #  ( A : )
        self.confidence_thresholds = {
            'high': 0.90,      # , ( 0.05)
            'medium': 0.70,    # , ( 0.10)
            'low': 0.50        # , ( 0.10)
        }

        # Blazer  (, Recall)
        self.blazer_special_threshold = 0.35  # Blazer

        #  ()
        self.exclusive_groups = [
            {'pants', 'shorts'},  # vs
            {'dress', 'skirt', 'pants'},  # vs  vs
            {'tee', 'blouse', 'blazer'},   # T vs  vs
        ]

        #  ()
        self.confusion_matrix = {
            'pants': {'shorts': 0.7, 'skirt': 0.3, 'dress': 0.2},
            'shorts': {'pants': 0.7, 'skirt': 0.2},
            'skirt': {'dress': 0.5, 'pants': 0.3, 'shorts': 0.2},
            'dress': {'skirt': 0.5, 'blouse': 0.3},
            'tee': {'blouse': 0.4, 'dress': 0.2},
            'blouse': {'tee': 0.4, 'dress': 0.3, 'blazer': 0.6},  # Blazer
            'blazer': {'blouse': 0.6, 'dress': 0.2, 'tee': 0.3}  # Blazer
        }

        #
        self.known_error_patterns = {
            # (, ) ->
            ('tee', 'pants'): 'pants',
            ('tee', 'jeans'): 'pants',
            ('tee', 'trousers'): 'pants',
            ('skirt', 'pants'): 'pants',
            ('dress', 'pants'): 'pants',
            ('blouse', 'shorts'): 'shorts',
            ('tee', 'shorts'): 'shorts',
            # Blazer
            ('blouse', 'blazer'): 'blazer',
            ('blouse', 'jacket'): 'blazer',
            ('tee', 'blazer'): 'blazer',
            ('dress', 'blazer'): 'blazer',
        }

    def validate_classification(
        self,
        predicted_category: str,
        confidence: float,
        all_scores: Dict[str, float],
        text_info: str = "",
        image_path: str = ""
    ) -> Tuple[str, float, Dict]:
        """


        Args:
            predicted_category: 
            confidence: 
            all_scores: 
            text_info: 
            image_path: 

        Returns:
            (final_category, final_confidence, validation_info)
        """
        validation_info = {
            'original_prediction': predicted_category,
            'original_confidence': confidence,
            'validation_level': None,
            'corrections': [],
            'warnings': []
        }

        # 1.
        if confidence >= self.confidence_thresholds['high']:
            validation_level = 'high'
        elif confidence >= self.confidence_thresholds['medium']:
            validation_level = 'medium'
        else:
            validation_level = 'low'

        validation_info['validation_level'] = validation_level

        # 2.
        if validation_level == 'high':
            correction = self._check_high_confidence_errors(
                predicted_category, text_info, image_path
            )
            if correction:
                validation_info['corrections'].append({
                    'type': 'high_confidence_error',
                    'reason': correction['reason'],
                    'suggested_category': correction['category']
                })
                predicted_category = correction['category']
                confidence = all_scores.get(
                    predicted_category, confidence * 0.9)

        # 3.
        if validation_level in ['medium', 'low']:
            # -
            text_correction = self._check_text_category_consistency(
                predicted_category, text_info, all_scores
            )
            if text_correction:
                validation_info['corrections'].append({
                    'type': 'text_consistency',
                    'reason': text_correction['reason'],
                    'suggested_category': text_correction['category']
                })
                predicted_category = text_correction['category']
                confidence = all_scores.get(
                    predicted_category, confidence * 1.1)

            # -
            path_correction = self._check_path_category_consistency(
                predicted_category, image_path, all_scores
            )
            if path_correction:
                validation_info['corrections'].append({
                    'type': 'path_consistency',
                    'reason': path_correction['reason'],
                    'suggested_category': path_correction['category']
                })
                predicted_category = path_correction['category']
                confidence = all_scores.get(
                    predicted_category, confidence * 1.1)

        # 4. Blazer  ()
        blazer_correction = self._check_blazer_special_case(
            predicted_category, all_scores, text_info, image_path
        )
        if blazer_correction:
            validation_info['corrections'].append({
                'type': 'blazer_special_detection',
                'reason': blazer_correction['reason'],
                'suggested_category': blazer_correction['category']
            })
            predicted_category = blazer_correction['category']
            confidence = all_scores.get('blazer', confidence)

        # 5.  ()
        competing_categories = self._find_competing_categories(
            predicted_category, all_scores, threshold=0.15
        )
        if competing_categories:
            validation_info['warnings'].append({
                'type': 'competing_categories',
                'categories': competing_categories,
                'message': f"Close scores with: {', '.join(competing_categories)}"
            })

        # 6.  ()
        if confidence < self.confidence_thresholds['low']:
            validation_info['warnings'].append({
                'type': 'low_confidence',
                'message': f"Very low confidence ({confidence:.3f}), consider manual review"
            })

        #  [0, 1]
        final_confidence = max(0.0, min(1.0, confidence))

        return predicted_category, final_confidence, validation_info

    def _check_blazer_special_case(
        self,
        predicted_category: str,
        all_scores: Dict[str, float],
        text: str,
        path: str
    ) -> Optional[Dict]:
        """
        Blazer 
         Blazer  (4.38%)  Blouse,
         Recall
        """
        text_lower = text.lower()
        path_lower = path.lower()

        #  Blazer
        strong_blazer_keywords = ['blazer', 'jacket',
                                  'vest', 'waistcoat', 'cardigan', 'coat']
        #  Blazer
        weak_blazer_keywords = ['formal', 'business',
                                'office', 'professional', 'suit']

        #  Blazer
        blazer_score = all_scores.get('blazer', 0.0)
        current_score = all_scores.get(predicted_category.lower(), 0.0)

        #  1:  Blazer , Blazer
        has_strong_keyword = any(
            kw in text_lower for kw in strong_blazer_keywords)
        if has_strong_keyword and predicted_category.lower() != 'blazer':
            #  Blazer ,
            if blazer_score >= self.blazer_special_threshold:
                return {
                    'category': 'blazer',
                    'reason': f'Strong blazer keyword detected (score: {blazer_score:.3f})'
                }

        #  2:  blazer, Blazer
        if 'blazer' in path_lower and predicted_category.lower() != 'blazer':
            if blazer_score >= self.blazer_special_threshold:
                return {
                    'category': 'blazer',
                    'reason': f'Blazer in path (score: {blazer_score:.3f})'
                }

        #  3: Blazer ,
        has_weak_keyword = any(kw in text_lower for kw in weak_blazer_keywords)
        score_diff = current_score - blazer_score
        if has_weak_keyword and predicted_category.lower() in ['blouse', 'tee', 'dress']:
            if score_diff < 0.15 and blazer_score >= 0.25:  # Blazer
                return {
                    'category': 'blazer',
                    'reason': f'Close scores with blazer hint (diff: {score_diff:.3f})'
                }

        #  4:  Blouse, Blazer
        if predicted_category.lower() == 'blouse' and blazer_score >= 0.30:
            if score_diff < 0.20:  # Blouse  Blazer
                #  Blazer
                has_any_blazer_hint = (
                    has_strong_keyword or
                    has_weak_keyword or
                    'blazer' in path_lower or
                    'jacket' in path_lower
                )
                if has_any_blazer_hint:
                    return {
                        'category': 'blazer',
                        'reason': f'Blouse-Blazer confusion resolved (blazer: {blazer_score:.3f}, blouse: {current_score:.3f})'
                    }

        return None

    def _check_high_confidence_errors(
        self,
        category: str,
        text: str,
        path: str
    ) -> Optional[Dict]:
        """"""
        text_lower = text.lower()
        path_lower = path.lower()

        #
        for (error_cat, keyword), correct_cat in self.known_error_patterns.items():
            if category == error_cat and keyword in text_lower:
                return {
                    'category': correct_cat,
                    'reason': f"Detected '{keyword}' in text but classified as '{error_cat}'"
                }

        #
        strong_signals = {
            'pants': ['pants', 'jeans', 'trousers', 'chinos'],
            'shorts': ['shorts', 'bermuda'],
            'skirt': ['skirt'],
            'dress': ['dress', 'gown'],
        }

        for correct_cat, keywords in strong_signals.items():
            if correct_cat != category:
                for keyword in keywords:
                    if keyword in text_lower or keyword in path_lower:
                        return {
                            'category': correct_cat,
                            'reason': f"Strong signal '{keyword}' contradicts '{category}'"
                        }

        return None

    def _check_text_category_consistency(
        self,
        category: str,
        text: str,
        all_scores: Dict[str, float]
    ) -> Optional[Dict]:
        """"""
        if not text:
            return None

        text_lower = text.lower()

        #
        category_keywords = {
            'pants': ['pants', 'jeans', 'trousers', 'chinos', 'slacks'],
            'shorts': ['shorts', 'bermuda'],
            'skirt': ['skirt', 'mini-skirt', 'midi-skirt'],
            'tee': ['t-shirt', 'tee', 'tank', 'cami'],
            'blouse': ['blouse', 'shirt', 'tunic'],
            'dress': ['dress', 'gown', 'frock'],
            'blazer': ['blazer', 'jacket', 'coat']
        }

        #
        for other_cat, keywords in category_keywords.items():
            if other_cat != category:
                for keyword in keywords:
                    if keyword in text_lower:
                        #
                        other_score = all_scores.get(other_cat, 0.0)
                        current_score = all_scores.get(category, 1.0)

                        # ,
                        if other_score > 0.2:
                            return {
                                'category': other_cat,
                                'reason': f"Text contains '{keyword}' indicating '{other_cat}'"
                            }

        return None

    def _check_path_category_consistency(
        self,
        category: str,
        path: str,
        all_scores: Dict[str, float]
    ) -> Optional[Dict]:
        """"""
        if not path:
            return None

        path_lower = path.lower()

        #
        path_patterns = {
            'pants': [r'pant', r'jean', r'trouser'],
            'shorts': [r'short'],
            'skirt': [r'skirt'],
            'dress': [r'dress', r'gown'],
            'tee': [r'tee', r't-shirt'],
            'blouse': [r'blouse'],
            'blazer': [r'blazer', r'jacket']
        }

        for other_cat, patterns in path_patterns.items():
            if other_cat != category:
                for pattern in patterns:
                    if re.search(pattern, path_lower):
                        other_score = all_scores.get(other_cat, 0.0)
                        if other_score > 0.15:
                            return {
                                'category': other_cat,
                                'reason': f"Path contains pattern '{pattern}' indicating '{other_cat}'"
                            }

        return None

    def _find_competing_categories(
        self,
        predicted_category: str,
        all_scores: Dict[str, float],
        threshold: float = 0.15
    ) -> List[str]:
        """"""
        predicted_score = all_scores.get(predicted_category, 0.0)
        competing = []

        for cat, score in all_scores.items():
            if cat != predicted_category:
                #
                if abs(score - predicted_score) < threshold:
                    competing.append(f"{cat}({score:.3f})")

        return competing

    def batch_validate(
        self,
        predictions: List[Tuple[str, float, Dict[str, float]]],
        text_infos: List[str] = None,
        image_paths: List[str] = None
    ) -> List[Tuple[str, float, Dict]]:
        """"""
        if text_infos is None:
            text_infos = [""] * len(predictions)
        if image_paths is None:
            image_paths = [""] * len(predictions)

        validated_results = []
        for (cat, conf, scores), text, path in zip(predictions, text_infos, image_paths):
            validated_cat, validated_conf, info = self.validate_classification(
                cat, conf, scores, text, path
            )
            validated_results.append((validated_cat, validated_conf, info))

        return validated_results

    def get_validation_statistics(
        self,
        validation_results: List[Tuple[str, float, Dict]]
    ) -> Dict:
        """"""
        total = len(validation_results)
        corrections = 0
        warnings = 0

        confidence_dist = Counter()
        correction_types = Counter()

        for _, _, info in validation_results:
            #
            confidence_dist[info['validation_level']] += 1

            #
            if info['corrections']:
                corrections += 1
                for correction in info['corrections']:
                    correction_types[correction['type']] += 1

            #
            if info['warnings']:
                warnings += 1

        return {
            'total_validated': total,
            'total_corrections': corrections,
            'correction_rate': corrections / total if total > 0 else 0,
            'total_warnings': warnings,
            'confidence_distribution': dict(confidence_dist),
            'correction_types': dict(correction_types)
        }


def test_smart_validator():
    """"""
    categories = ['blazer', 'blouse', 'dress',
                  'skirt', 'tee', 'pants', 'shorts']
    validator = SmartValidator(categories)

    print("="*80)
    print("Smart Validator Initialized!")
    print("="*80)
    print(f"Categories: {categories}")
    print(f"Confidence Thresholds: {validator.confidence_thresholds}")
    print(f"\nReady to validate classifications and reduce errors! ")

    #
    test_cases = [
        {
            'predicted': 'tee',
            'confidence': 0.95,
            'scores': {'tee': 0.95, 'blouse': 0.03, 'pants': 0.02},
            'text': 'Blue denim jeans for men',
            'path': 'datasets/images/pants_001.jpg'
        },
        {
            'predicted': 'skirt',
            'confidence': 0.65,
            'scores': {'skirt': 0.65, 'pants': 0.25, 'dress': 0.10},
            'text': 'Cotton trousers',
            'path': 'images/bottom_wear.jpg'
        }
    ]

    print("\n" + "="*80)
    print("Testing Sample Cases:")
    print("="*80)

    for i, case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"  Original: {case['predicted']} ({case['confidence']:.3f})")
        print(f"  Text: {case['text']}")

        final_cat, final_conf, info = validator.validate_classification(
            case['predicted'],
            case['confidence'],
            case['scores'],
            case['text'],
            case['path']
        )

        print(f"  Validated: {final_cat} ({final_conf:.3f})")
        if info['corrections']:
            print(f"    Corrections made: {len(info['corrections'])}")
            for corr in info['corrections']:
                print(f"     - {corr['type']}: {corr['reason']}")
        else:
            print(f"   No corrections needed")

    return validator


if __name__ == "__main__":
    validator = test_smart_validator()

    print("\n" + "="*80)
    print("Example Usage:")
    print("="*80)
    print("""
from smart_validator import SmartValidator

# 
validator = SmartValidator(categories)

# 
final_category, final_confidence, validation_info = validator.validate_classification(
    predicted_category='tee',
    confidence=0.95,
    all_scores={'tee': 0.95, 'pants': 0.03, 'blouse': 0.02},
    text_info='Blue denim jeans',
    image_path='datasets/images/pants_001.jpg'
)

# 
if validation_info['corrections']:
    print(f"Corrected: {validation_info['original_prediction']} -> {final_category}")
    for correction in validation_info['corrections']:
        print(f"  Reason: {correction['reason']}")
    """)
