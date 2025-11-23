#!/usr/bin/env python3
"""
IEEE Academic Paper Visualization Generator
Generates publication-quality figures for the AI Wardrobe Assistant research paper.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.patches as mpatches
from matplotlib import rcParams

# IEEE Publication Style Settings
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 11
rcParams['axes.titlesize'] = 12
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 9
rcParams['figure.titlesize'] = 13

# Professional color palette
COLORS = {
    'primary': '#1E88E5',      # Blue
    'secondary': '#FFA726',    # Orange
    'success': '#66BB6A',      # Green
    'warning': '#EF5350',      # Red
    'neutral': '#78909C',      # Gray
    'accent': '#AB47BC',       # Purple
    'highlight': '#26C6DA'     # Cyan
}

# Create output directory
OUTPUT_DIR = Path('results/ieee_report')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Data from evaluation results
CATEGORIES = ['Blazer', 'Blouse', 'Dress', 'Skirt', 'Tee', 'Pants', 'Shorts']
PRECISION = [1.000, 0.737, 0.750, 0.600, 0.588, 0.706, 1.000]
RECALL = [0.500, 1.000, 0.643, 0.643, 0.714, 0.857, 0.786]
F1_SCORES = [0.667, 0.848, 0.692, 0.621, 0.645, 0.774, 0.880]
SUPPORT = [14, 14, 14, 14, 14, 14, 14]

# Confusion Matrix (7x7 for 7 categories)
CONFUSION_MATRIX = np.array([
    [7, 0, 0, 0, 0, 0, 0],      # Blazer
    [0, 14, 0, 0, 0, 0, 0],     # Blouse
    [0, 4, 9, 0, 1, 0, 0],      # Dress
    [0, 0, 1, 9, 2, 2, 0],      # Skirt
    [0, 0, 1, 1, 10, 2, 0],     # Tee
    [0, 0, 0, 0, 2, 12, 0],     # Pants
    [0, 0, 0, 0, 3, 0, 11]      # Shorts
])


def figure1_system_evolution():
    """
    Figure 1: System Evolution Comparison
    Shows improvement from Deliverable 2 to current system
    """
    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)

    systems = ['Deliverable 2\n(ResNet50)', 'Deliverable 3\n(Ensemble)']
    accuracies = [56.6, 73.47]
    colors_list = [COLORS['secondary'], COLORS['primary']]

    bars = ax.bar(systems, accuracies, color=colors_list, alpha=0.8,
                  edgecolor='black', linewidth=1.5, width=0.6)

    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

        # Add improvement annotation for second bar
        if i == 1:
            improvement = acc - accuracies[0]
            ax.annotate(f'+{improvement:.2f}%\nimprovement',
                        xy=(bar.get_x() + bar.get_width()/2., height/2),
                        ha='center', va='center',
                        fontsize=10, color='white', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5',
                                  facecolor=COLORS['success'], alpha=0.8))

    ax.set_ylabel('Classification Accuracy (%)', fontweight='bold')
    ax.set_title('System Performance Evolution', fontweight='bold', pad=15)
    ax.set_ylim(0, 85)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add baseline reference line
    ax.axhline(y=56.6, color=COLORS['neutral'], linestyle='--',
               linewidth=1, alpha=0.5, label='Baseline')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_system_evolution.png',
                dpi=300, bbox_inches='tight')
    print(f"✅ Generated: fig1_system_evolution.png")
    plt.close()


def figure2_per_class_performance():
    """
    Figure 2: Per-Class Performance Analysis (Grouped Bar Chart)
    """
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    x = np.arange(len(CATEGORIES))
    width = 0.25

    bars1 = ax.bar(x - width, PRECISION, width, label='Precision',
                   color=COLORS['primary'], alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, RECALL, width, label='Recall',
                   color=COLORS['secondary'], alpha=0.8, edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + width, F1_SCORES, width, label='F1-Score',
                   color=COLORS['success'], alpha=0.8, edgecolor='black', linewidth=0.5)

    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=7, rotation=0)

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    ax.set_xlabel('Garment Category', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIES, rotation=45, ha='right')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add average line
    avg_f1 = np.mean(F1_SCORES)
    ax.axhline(y=avg_f1, color=COLORS['neutral'], linestyle='--',
               linewidth=1.5, alpha=0.6, label=f'Avg F1: {avg_f1:.3f}')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_per_class_performance.png',
                dpi=300, bbox_inches='tight')
    print(f"✅ Generated: fig2_per_class_performance.png")
    plt.close()


def figure3_radar_chart():
    """
    Figure 3: Radar Chart for Performance Metrics
    """
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300,
                           subplot_kw=dict(projection='polar'))

    angles = np.linspace(0, 2 * np.pi, len(CATEGORIES),
                         endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    precision_plot = PRECISION + PRECISION[:1]
    recall_plot = RECALL + RECALL[:1]
    f1_plot = F1_SCORES + F1_SCORES[:1]

    ax.plot(angles, precision_plot, 'o-', linewidth=2,
            label='Precision', color=COLORS['primary'])
    ax.fill(angles, precision_plot, alpha=0.15, color=COLORS['primary'])

    ax.plot(angles, recall_plot, 's-', linewidth=2,
            label='Recall', color=COLORS['secondary'])
    ax.fill(angles, recall_plot, alpha=0.15, color=COLORS['secondary'])

    ax.plot(angles, f1_plot, '^-', linewidth=2,
            label='F1-Score', color=COLORS['success'])
    ax.fill(angles, f1_plot, alpha=0.15, color=COLORS['success'])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(CATEGORIES, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)

    ax.set_title('Performance Metrics Across Categories',
                 fontweight='bold', pad=20, fontsize=13)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), framealpha=0.9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_radar_chart.png',
                dpi=300, bbox_inches='tight')
    print(f"✅ Generated: fig3_radar_chart.png")
    plt.close()


def figure4_confusion_matrix():
    """
    Figure 4: Confusion Matrix Heatmap
    """
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)

    # Normalize by row (true labels)
    cm_normalized = CONFUSION_MATRIX.astype(
        'float') / CONFUSION_MATRIX.sum(axis=1)[:, np.newaxis]

    # Create heatmap
    sns.heatmap(cm_normalized, annot=CONFUSION_MATRIX, fmt='d',
                cmap='Blues', cbar_kws={'label': 'Normalized Count'},
                xticklabels=CATEGORIES, yticklabels=CATEGORIES,
                linewidths=0.5, linecolor='gray', ax=ax,
                vmin=0, vmax=1)

    ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=11)
    ax.set_ylabel('True Label', fontweight='bold', fontsize=11)
    ax.set_title('Confusion Matrix (Test Set: 98 Images)',
                 fontweight='bold', pad=15, fontsize=12)

    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_confusion_matrix.png',
                dpi=300, bbox_inches='tight')
    print(f"✅ Generated: fig4_confusion_matrix.png")
    plt.close()


def figure5_component_contribution():
    """
    Figure 5: Ensemble Component Contribution Analysis
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Pie Chart
    components = ['CLIP ViT-B/32', 'Keyword Matching', 'Path Analysis']
    contributions = [95, 3, 2]
    colors_pie = [COLORS['primary'], COLORS['secondary'], COLORS['accent']]
    explode = (0.05, 0.05, 0.05)

    wedges, texts, autotexts = ax1.pie(contributions, labels=components, autopct='%1.1f%%',
                                       startangle=90, colors=colors_pie, explode=explode,
                                       textprops={'fontsize': 10,
                                                  'fontweight': 'bold'},
                                       wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)

    ax1.set_title('Ensemble Weight Distribution', fontweight='bold', pad=15)

    # Bar Chart with comparison
    ax2.bar(components, contributions, color=colors_pie, alpha=0.8,
            edgecolor='black', linewidth=1.5)

    # Add value labels
    for i, (comp, val) in enumerate(zip(components, contributions)):
        ax2.text(i, val + 2, f'{val}%', ha='center', va='bottom',
                 fontweight='bold', fontsize=10)

    ax2.set_ylabel('Contribution Weight (%)', fontweight='bold')
    ax2.set_title('Component Contribution Breakdown',
                  fontweight='bold', pad=15)
    ax2.set_ylim(0, 105)
    ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    plt.setp(ax2.get_xticklabels(), rotation=25, ha='right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig5_component_contribution.png',
                dpi=300, bbox_inches='tight')
    print(f"✅ Generated: fig5_component_contribution.png")
    plt.close()


def figure6_performance_summary():
    """
    Figure 6: Overall Performance Summary Table (as image)
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.axis('tight')
    ax.axis('off')

    # Create summary table data
    table_data = []
    table_data.append(
        ['Category', 'Precision', 'Recall', 'F1-Score', 'Support'])

    for i, cat in enumerate(CATEGORIES):
        table_data.append([
            cat,
            f'{PRECISION[i]:.3f}',
            f'{RECALL[i]:.3f}',
            f'{F1_SCORES[i]:.3f}',
            str(SUPPORT[i])
        ])

    # Add average row
    table_data.append([
        'Macro Avg',
        f'{np.mean(PRECISION):.3f}',
        f'{np.mean(RECALL):.3f}',
        f'{np.mean(F1_SCORES):.3f}',
        str(sum(SUPPORT))
    ])

    # Add weighted average
    weighted_f1 = np.average(F1_SCORES, weights=SUPPORT)
    table_data.append([
        'Weighted Avg',
        f'{np.average(PRECISION, weights=SUPPORT):.3f}',
        f'{np.average(RECALL, weights=SUPPORT):.3f}',
        f'{weighted_f1:.3f}',
        str(sum(SUPPORT))
    ])

    # Add overall accuracy
    table_data.append(['Overall Accuracy', '', '', '73.47%', '98'])

    # Create table
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.18, 0.18, 0.18, 0.15])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)

    # Style header row
    for i in range(5):
        cell = table[(0, i)]
        cell.set_facecolor(COLORS['primary'])
        cell.set_text_props(weight='bold', color='white')

    # Style data rows
    for i in range(1, len(table_data)):
        for j in range(5):
            cell = table[(i, j)]
            if i == len(table_data) - 1:  # Overall accuracy row
                cell.set_facecolor(COLORS['success'])
                cell.set_text_props(weight='bold', color='white')
            elif i >= len(CATEGORIES) + 1:  # Average rows
                cell.set_facecolor('#E3F2FD')
                cell.set_text_props(weight='bold')
            else:
                cell.set_facecolor('#F5F5F5' if i % 2 == 0 else 'white')

    ax.set_title('Classification Performance Summary',
                 fontweight='bold', pad=20, fontsize=13)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig6_performance_summary.png',
                dpi=300, bbox_inches='tight')
    print(f"✅ Generated: fig6_performance_summary.png")
    plt.close()


def figure7_model_architecture():
    """
    Figure 7: System Architecture Diagram (Simplified Block Diagram)
    """
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Define boxes
    def draw_box(ax, x, y, w, h, text, color, text_color='black'):
        rect = mpatches.FancyBboxPatch((x, y), w, h,
                                       boxstyle="round,pad=0.1",
                                       edgecolor='black', facecolor=color,
                                       linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=10, fontweight='bold', color=text_color,
                wrap=True)

    def draw_arrow(ax, x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Input layer
    draw_box(ax, 0.5, 6, 1.5, 1, 'Input\nImage', COLORS['neutral'], 'white')

    # Three processing streams
    y_positions = [5.5, 3.5, 1.5]
    stream_names = [
        'CLIP\nViT-B/32\n(95%)', 'Keyword\nMatch\n(3%)', 'Path\nAnalysis\n(2%)']
    stream_colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent']]

    for i, (y, name, color) in enumerate(zip(y_positions, stream_names, stream_colors)):
        # Processing blocks
        draw_box(ax, 2.5, y, 1.5, 0.8, name, color, 'white')
        draw_arrow(ax, 2, 6.5, 2.5, y + 0.4)

        # Feature extraction
        draw_box(ax, 4.5, y, 1.5, 0.8, 'Features', '#E3F2FD', 'black')
        draw_arrow(ax, 4, y + 0.4, 4.5, y + 0.4)

        # Connect to ensemble
        draw_arrow(ax, 6, y + 0.4, 6.5, 4)

    # Ensemble layer
    draw_box(ax, 6.5, 3.5, 1.5, 1.2, 'Ensemble\nClassifier',
             COLORS['success'], 'white')

    # Output
    draw_box(ax, 8.5, 3.5, 1.2, 1.2, 'Category\nPrediction',
             COLORS['warning'], 'white')
    draw_arrow(ax, 8, 4.1, 8.5, 4.1)

    # Title
    ax.text(5, 7.5, 'Ensemble Classification System Architecture',
            ha='center', fontsize=13, fontweight='bold')

    # Add accuracy annotation
    ax.text(9.1, 2.8, '73.47%\nAccuracy', ha='center', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig7_architecture.png',
                dpi=300, bbox_inches='tight')
    print(f"✅ Generated: fig7_architecture.png")
    plt.close()


def generate_all_figures():
    """Generate all IEEE publication figures"""
    print("\n" + "="*60)
    print("IEEE ACADEMIC VISUALIZATION GENERATOR")
    print("="*60)
    print(f"\nOutput directory: {OUTPUT_DIR.absolute()}\n")

    try:
        figure1_system_evolution()
        figure2_per_class_performance()
        figure3_radar_chart()
        figure4_confusion_matrix()
        figure5_component_contribution()
        figure6_performance_summary()
        figure7_model_architecture()

        print("\n" + "="*60)
        print("✅ ALL FIGURES GENERATED SUCCESSFULLY!")
        print("="*60)
        print(f"\nGenerated 7 publication-quality figures:")
        print(f"  📊 fig1_system_evolution.png")
        print(f"  📊 fig2_per_class_performance.png")
        print(f"  📊 fig3_radar_chart.png")
        print(f"  📊 fig4_confusion_matrix.png")
        print(f"  📊 fig5_component_contribution.png")
        print(f"  📊 fig6_performance_summary.png")
        print(f"  📊 fig7_architecture.png")
        print(f"\nLocation: {OUTPUT_DIR.absolute()}")
        print("\nAll figures are:")
        print("  ✅ 300 DPI high resolution")
        print("  ✅ IEEE publication style")
        print("  ✅ Professional color scheme")
        print("  ✅ Properly labeled and annotated")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n❌ Error generating figures: {e}")
        raise


if __name__ == "__main__":
    generate_all_figures()
