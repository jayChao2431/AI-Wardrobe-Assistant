"""
Recommendation Explainer Module
Provides detailed explanations for why items are recommended
"""
import numpy as np
from typing import Dict, List, Tuple
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


class RecommendationExplainer:
    """Generate human-readable explanations for recommendations"""

    # Style keywords for analysis
    STYLE_KEYWORDS = {
        'casual': ['casual', 'relaxed', 'everyday', 'comfortable', 't-shirt', 'tee', 'jeans'],
        'formal': ['formal', 'business', 'professional', 'office', 'blazer', 'suit', 'dress shirt'],
        'elegant': ['elegant', 'sophisticated', 'chic', 'classy', 'refined', 'dress', 'evening'],
        'sporty': ['sport', 'athletic', 'active', 'gym', 'workout', 'running', 'joggers'],
        'vintage': ['vintage', 'retro', 'classic', 'traditional', 'old school'],
        'modern': ['modern', 'contemporary', 'trendy', 'fashion', 'stylish'],
        'minimalist': ['minimal', 'simple', 'basic', 'clean', 'plain'],
        'bohemian': ['boho', 'bohemian', 'ethnic', 'hippie', 'flowy'],
    }

    # Color keywords
    COLOR_KEYWORDS = {
        'black': ['black', 'ebony', 'dark'],
        'white': ['white', 'ivory', 'cream', 'off-white'],
        'blue': ['blue', 'navy', 'denim', 'azure', 'cobalt'],
        'red': ['red', 'crimson', 'scarlet', 'burgundy', 'wine'],
        'green': ['green', 'olive', 'emerald', 'forest'],
        'yellow': ['yellow', 'gold', 'mustard', 'amber'],
        'pink': ['pink', 'rose', 'blush', 'coral'],
        'gray': ['gray', 'grey', 'silver', 'charcoal'],
        'brown': ['brown', 'tan', 'beige', 'khaki', 'camel'],
        'purple': ['purple', 'violet', 'lavender', 'plum'],
    }

    # Material keywords
    MATERIAL_KEYWORDS = {
        'cotton': ['cotton', '100% cotton'],
        'denim': ['denim', 'jean'],
        'silk': ['silk', 'satin'],
        'leather': ['leather', 'faux leather', 'pu leather'],
        'wool': ['wool', 'cashmere', 'knit'],
        'polyester': ['polyester', 'synthetic'],
        'linen': ['linen'],
    }

    def __init__(self):
        pass

    def extract_features(self, meta: Dict) -> Dict[str, List[str]]:
        """Extract style, color, and material features from metadata"""
        text = (meta.get('title', '') + ' ' +
                meta.get('description', '')).lower()

        features = {
            'styles': [],
            'colors': [],
            'materials': []
        }

        # Extract styles
        for style, keywords in self.STYLE_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                features['styles'].append(style)

        # Extract colors
        for color, keywords in self.COLOR_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                features['colors'].append(color)

        # Extract materials
        for material, keywords in self.MATERIAL_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                features['materials'].append(material)

        return features

    def generate_explanation(
        self,
        query_meta: Dict,
        rec_meta: Dict,
        similarity: float,
        query_category: str,
        outfit_mode: bool = False
    ) -> Dict[str, any]:
        """
        Generate detailed explanation for a recommendation

        Returns:
            Dict with explanation components:
            - main_reason: Primary reason for recommendation
            - style_match: Style compatibility explanation
            - color_match: Color coordination explanation
            - occasion: Suitable occasions
            - confidence: Explanation confidence (0-1)
        """
        query_features = self.extract_features(query_meta)
        rec_features = self.extract_features(rec_meta)

        explanation = {
            'main_reason': '',
            'style_match': '',
            'color_match': '',
            'material_info': '',
            'occasion': '',
            'confidence': self._calculate_confidence(similarity, query_features, rec_features)
        }

        # Main reason
        if outfit_mode:
            explanation['main_reason'] = self._get_outfit_reason(
                query_category, rec_meta.get('category', 'Unknown')
            )
        else:
            explanation['main_reason'] = self._get_similarity_reason(
                similarity)

        # Style match
        explanation['style_match'] = self._analyze_style_match(
            query_features['styles'], rec_features['styles']
        )

        # Color match
        explanation['color_match'] = self._analyze_color_match(
            query_features['colors'], rec_features['colors'], outfit_mode
        )

        # Material info
        if rec_features['materials']:
            materials_text = ', '.join(rec_features['materials'])
            explanation['material_info'] = f"Made from {materials_text}"

        # Occasion
        explanation['occasion'] = self._suggest_occasion(
            rec_features['styles'], query_features['styles'], outfit_mode
        )

        return explanation

    def _get_outfit_reason(self, query_cat: str, rec_cat: str) -> str:
        """Generate reason for outfit matching"""
        TOPS = ['Blazer', 'Blouse', 'Tee']
        BOTTOMS = ['Skirt', 'Pants', 'Shorts']

        if query_cat in TOPS and rec_cat in BOTTOMS:
            return f"Perfect pairing: {query_cat} (top) matches well with {rec_cat} (bottom)"
        elif query_cat in BOTTOMS and rec_cat in TOPS:
            return f"Complete outfit: {rec_cat} (top) complements your {query_cat} (bottom)"
        elif query_cat == 'Dress':
            return "Similar complete outfit style for coordinated look"
        else:
            return "Complementary item for outfit coordination"

    def _get_similarity_reason(self, similarity: float) -> str:
        """Generate reason based on similarity score"""
        if similarity > 0.9:
            return "Extremely similar style and visual features"
        elif similarity > 0.8:
            return "Highly similar design and aesthetic"
        elif similarity > 0.7:
            return "Notable visual similarities detected"
        else:
            return "Some visual resemblance in design"

    def _analyze_style_match(self, query_styles: List[str], rec_styles: List[str]) -> str:
        """Analyze style compatibility"""
        if not query_styles or not rec_styles:
            return "Style information not available"

        common_styles = set(query_styles) & set(rec_styles)

        if common_styles:
            styles_text = ', '.join(common_styles)
            return f"Matches your {styles_text} style preference"
        else:
            # Check for complementary styles
            if 'formal' in query_styles and 'elegant' in rec_styles:
                return "Elegant style complements your formal aesthetic"
            elif 'casual' in query_styles and 'sporty' in rec_styles:
                return "Sporty style aligns with casual vibe"
            else:
                return f"Different but compatible: {rec_styles[0]} style"

    def _analyze_color_match(self, query_colors: List[str], rec_colors: List[str], outfit_mode: bool) -> str:
        """Analyze color coordination"""
        if not rec_colors:
            return ""

        # Color matching rules
        COMPLEMENTARY_COLORS = {
            'black': ['white', 'gray', 'red', 'pink', 'gold'],
            'white': ['black', 'blue', 'navy', 'gray', 'any color'],
            'blue': ['white', 'brown', 'beige', 'gray', 'yellow'],
            'navy': ['white', 'beige', 'brown', 'red'],
            'gray': ['white', 'black', 'pink', 'blue', 'yellow'],
            'beige': ['white', 'brown', 'navy', 'blue'],
            'brown': ['beige', 'white', 'blue', 'green'],
        }

        rec_color = rec_colors[0] if rec_colors else None

        if not query_colors:
            return f"Features {rec_color} color"

        query_color = query_colors[0]

        if query_color == rec_color:
            if outfit_mode:
                return f"Matching {rec_color} creates cohesive look"
            else:
                return f"Same {rec_color} color palette"
        else:
            # Check if colors complement each other
            if rec_color in COMPLEMENTARY_COLORS.get(query_color, []):
                return f"{rec_color.capitalize()} pairs beautifully with your {query_color} item"
            else:
                return f"Contrasting {rec_color} for visual interest"

    def _suggest_occasion(self, rec_styles: List[str], query_styles: List[str], outfit_mode: bool) -> str:
        """Suggest suitable occasions"""
        all_styles = set(rec_styles + query_styles)

        if 'formal' in all_styles or 'elegant' in all_styles:
            return "Perfect for: Office, Business meetings, Formal events"
        elif 'casual' in all_styles:
            return "Perfect for: Daily wear, Weekend outings, Relaxed settings"
        elif 'sporty' in all_styles:
            return "Perfect for: Gym, Active lifestyle, Casual sports"
        elif 'elegant' in all_styles:
            return "Perfect for: Dinner dates, Parties, Special occasions"
        else:
            return "Versatile: Suitable for various occasions"

    def _calculate_confidence(self, similarity: float, query_features: Dict, rec_features: Dict) -> float:
        """Calculate explanation confidence based on available information"""
        confidence = similarity * 0.6  # Base confidence from similarity

        # Boost confidence if we have rich metadata
        if query_features['styles']:
            confidence += 0.1
        if query_features['colors']:
            confidence += 0.1
        if rec_features['styles']:
            confidence += 0.1
        if rec_features['colors']:
            confidence += 0.1

        return min(confidence, 1.0)

    def get_feature_comparison(self, query_meta: Dict, rec_meta: Dict) -> Dict[str, any]:
        """Generate side-by-side feature comparison"""
        query_features = self.extract_features(query_meta)
        rec_features = self.extract_features(rec_meta)

        return {
            'query': query_features,
            'recommendation': rec_features,
            'style_overlap': list(set(query_features['styles']) & set(rec_features['styles'])),
            'color_overlap': list(set(query_features['colors']) & set(rec_features['colors'])),
        }

    def visualize_feature_similarity(
        self,
        query_embedding: np.ndarray,
        rec_embedding: np.ndarray,
        save_path: str = None
    ) -> plt.Figure:
        """
        Visualize CLIP feature similarity as a heatmap

        Args:
            query_embedding: Query image CLIP embedding (512-dim)
            rec_embedding: Recommendation image CLIP embedding (512-dim)
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object
        """
        # Reshape embeddings for visualization
        # Group features into 16x32 blocks for better visualization
        query_reshaped = query_embedding[:512].reshape(16, 32)
        rec_reshaped = rec_embedding[:512].reshape(16, 32)

        # Calculate element-wise similarity
        similarity_map = query_reshaped * rec_reshaped

        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Query features
        im1 = axes[0].imshow(query_reshaped, cmap='viridis', aspect='auto')
        axes[0].set_title('Query Image Features',
                          fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Feature Dimension')
        axes[0].set_ylabel('Feature Group')
        plt.colorbar(im1, ax=axes[0], fraction=0.046)

        # Recommendation features
        im2 = axes[1].imshow(rec_reshaped, cmap='viridis', aspect='auto')
        axes[1].set_title('Recommended Image Features',
                          fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Feature Dimension')
        axes[1].set_ylabel('Feature Group')
        plt.colorbar(im2, ax=axes[1], fraction=0.046)

        # Similarity map
        im3 = axes[2].imshow(similarity_map, cmap='RdYlGn', aspect='auto')
        axes[2].set_title('Feature Similarity Heatmap',
                          fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Feature Dimension')
        axes[2].set_ylabel('Feature Group')
        plt.colorbar(im3, ax=axes[2], fraction=0.046, label='Similarity')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def generate_simple_similarity_bar(
        self,
        similarity_score: float,
        category_match: bool = False,
        gender_match: bool = False
    ) -> plt.Figure:
        """
        Generate a simple similarity visualization bar chart

        Args:
            similarity_score: Overall similarity (0-1)
            category_match: Whether categories match
            gender_match: Whether genders match

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(8, 3))

        categories = ['Visual\nSimilarity', 'Category\nMatch',
                      'Gender\nMatch', 'Overall\nScore']
        scores = [
            similarity_score,
            1.0 if category_match else 0.0,
            1.0 if gender_match else 0.0,
            (similarity_score + (1.0 if category_match else 0.0) +
             (1.0 if gender_match else 0.0)) / 3.0
        ]

        colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
        bars = ax.bar(categories, scores, color=colors,
                      alpha=0.8, edgecolor='black', linewidth=1.5)

        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.1%}',
                    ha='center', va='bottom', fontweight='bold', fontsize=11)

        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Match Score', fontsize=12, fontweight='bold')
        ax.set_title('Recommendation Match Analysis',
                     fontsize=14, fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Add threshold line
        ax.axhline(y=0.7, color='green', linestyle='--',
                   alpha=0.5, label='Good Match (70%)')
        ax.legend(loc='upper right', fontsize=9)

        plt.tight_layout()
        return fig
