"""
charts.py
Visualizations for the Travel Planner & Journal themed Visa System.
Color palette: Sage Green (#2D6A4F), Dusty Blue (#3D5A80), Terracotta (#E07A5F), and Muted Slate (#6C757D).
Clean white chart backgrounds on warm ivory canvas.
"""

import os
import tempfile
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "mpl_cache")

import matplotlib.pyplot as plt
import numpy as np

# Travel Planner Palette
TEXT_DARK = '#202938'
TEXT_MUTED = '#5A6578'
COLOR_SAGE = '#2D6A4F'       # Sage / Forest Green
COLOR_DUSTY_BLUE = '#3D5A80' # Dusty Navy/Blue
COLOR_CORAL = '#E07A5F'      # Soft Terracotta / Coral
COLOR_AMBER = '#D4A373'      # Warm Sand / Amber
COLOR_SLATE = '#6C757D'
COLOR_GRID = '#EDE8E1'       # Warm ivory grid
BG_WHITE = '#FFFFFF'


def plot_category_breakdown(category_scores: dict):
    """
    Renders a clean horizontal breakdown bar chart with warm travel planner styling.
    """
    categories = list(category_scores.keys())
    scores = list(category_scores.values())

    fig, ax = plt.subplots(figsize=(7, 3.2), facecolor=BG_WHITE)
    ax.set_facecolor(BG_WHITE)

    # Clean travel palette based on completion percentage
    colors = []
    for s in scores:
        if s >= 80:
            colors.append(COLOR_SAGE)        # Sage Green (strong)
        elif s >= 60:
            colors.append(COLOR_DUSTY_BLUE)  # Dusty Blue (moderate)
        else:
            colors.append(COLOR_CORAL)       # Terracotta (needs attention)

    y_pos = np.arange(len(categories))
    bars = ax.barh(y_pos, scores, color=colors, height=0.52, edgecolor='none')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=9.5, fontweight='bold', color=TEXT_DARK)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Readiness Percentage (%)", fontsize=9, fontweight='bold', color=TEXT_MUTED)
    ax.axvline(x=80, color=COLOR_DUSTY_BLUE, linestyle='--', linewidth=1.2, alpha=0.55, label='Travel Target (80%)')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#D6D0C7')
    ax.spines['bottom'].set_color('#D6D0C7')
    ax.grid(axis='x', linestyle='-', alpha=0.7, color=COLOR_GRID)

    for bar in bars:
        width = bar.get_width()
        ax.annotate(f"{width:.0f}%",
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(6, 0),
                    textcoords="offset points",
                    ha='left', va='center',
                    fontsize=9, fontweight='bold', color=TEXT_DARK)

    ax.legend(loc='lower right', frameon=True, facecolor=BG_WHITE, edgecolor='#D6D0C7', fontsize=8)
    plt.tight_layout()
    return fig


def plot_comparative_summary(readiness_score: float, ml_probability: float, baseline_avg: float = 75.0):
    """
    Renders a comparative bar chart with high-contrast text and warm travel tones.
    """
    fig, ax = plt.subplots(figsize=(6.5, 3.2), facecolor=BG_WHITE)
    ax.set_facecolor(BG_WHITE)

    labels = ['Document\nReadiness', 'Historical Profile\nMatch', 'Country\nAverage']
    values = [readiness_score, ml_probability, baseline_avg]
    colors = [COLOR_SAGE, COLOR_DUSTY_BLUE, COLOR_SLATE]

    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, values, color=colors, width=0.42, edgecolor='none')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9, fontweight='bold', color=TEXT_DARK)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Readiness / Likelihood (%)", fontsize=9, fontweight='bold', color=TEXT_MUTED)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#D6D0C7')
    ax.spines['bottom'].set_color('#D6D0C7')
    ax.grid(axis='y', linestyle='-', alpha=0.7, color=COLOR_GRID)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.0f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=9.5, fontweight='bold', color=TEXT_DARK)

    plt.tight_layout()
    return fig


def plot_feature_importance_chart(top_features: dict):
    """
    Plots the top profile factors with dark, legible labels and dusty blue bars.
    """
    features = list(top_features.keys())[::-1]
    weights = list(top_features.values())[::-1]

    cleaned_names = [
        f.replace("education_of_employee_", "Education: ")
         .replace("continent_", "Origin Region: ")
         .replace("has_job_experience_", "Work Experience: ")
         .replace("annual_wage", "Annual Income / Funding")
         .replace("no_of_employees", "Employer Scale")
         .replace("company_age", "Institution History")
        for f in features
    ]

    fig, ax = plt.subplots(figsize=(7, 3.4), facecolor=BG_WHITE)
    ax.set_facecolor(BG_WHITE)

    y_pos = np.arange(len(cleaned_names))
    bars = ax.barh(y_pos, weights, color=COLOR_DUSTY_BLUE, height=0.55, edgecolor='none')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(cleaned_names, fontsize=8.5, fontweight='bold', color=TEXT_DARK)
    ax.set_xlabel("Relative Factor Influence Weight", fontsize=8.5, fontweight='bold', color=TEXT_MUTED)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#D6D0C7')
    ax.spines['bottom'].set_color('#D6D0C7')
    ax.grid(axis='x', linestyle='-', alpha=0.7, color=COLOR_GRID)

    for bar in bars:
        width = bar.get_width()
        ax.annotate(f"{width:.3f}",
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0),
                    textcoords="offset points",
                    ha='left', va='center',
                    fontsize=8, fontweight='bold', color=TEXT_DARK)

    plt.tight_layout()
    return fig
