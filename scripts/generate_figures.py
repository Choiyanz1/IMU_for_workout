import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import json
from pathlib import Path

# Setup
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Color scheme - professional, print-friendly
COLORS = {
    'primary': '#2563eb',      # Blue
    'secondary': '#059669',    # Green  
    'accent': '#dc2626',       # Red
    'warning': '#d97706',      # Orange
    'neutral': '#64748b',      # Slate
    'bg': '#f8fafc',           # Light background
    'text': '#1e293b',         # Dark text
}

def save_fig(fig, name, dpi=300):
    """Save figure in multiple formats."""
    fig.savefig(FIG_DIR / f"{name}.png", dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(FIG_DIR / f"{name}.svg", bbox_inches='tight', facecolor='white')
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches='tight', facecolor='white')
    print(f"  Saved: {name}.png/.svg/.pdf")


# ============================================================================
# FIGURE 1: Window Size Optimization Curve
# ============================================================================
def plot_window_size_optimization():
    """Plot Rep F1 vs window_size for Causal RF configuration."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data from 04-baseline-comparison.md Table 2
    window_sizes = [50, 50, 100, 150, 100]
    n_estimators = [50, 100, 100, 100, 100]
    rep_f1 = [0.706, 0.723, 0.777, 0.778, 0.783]
    std = [0.091, 0.094, 0.057, 0.044, 0.064]
    labels = [
        "w=50, n=50\n(baseline)",
        "w=50, n=100",
        "w=100, n=100\n(optimal)",
        "w=150, n=100",
        "w=100, n=100\nsmooth=25"
    ]
    
    x_pos = np.arange(len(window_sizes))
    bars = ax.bar(x_pos, rep_f1, yerr=std, capsize=8, 
                   color=[COLORS['neutral'] if i < 2 else COLORS['primary'] if i == 2 else COLORS['secondary'] for i in range(5)],
                   edgecolor='white', linewidth=1.5, alpha=0.9)
    
    # Highlight optimal
    bars[2].set_edgecolor(COLORS['primary'])
    bars[2].set_linewidth(3)
    
    # Add value labels on bars
    for i, (bar, val, err) in enumerate(zip(bars, rep_f1, std)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + err + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold',
                color=COLORS['primary'] if i == 2 else COLORS['text'])
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Rep F1 Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_title('Causal RF Configuration Optimization\n(7-fold LOSO, 226 streams)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0.6, 0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add annotation for key finding
    ax.annotate('Key Finding: 1.0s window\n(+0.071 F1 improvement)', 
                xy=(2, 0.777), xytext=(3.5, 0.72),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=2),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#fef3c7', edgecolor=COLORS['warning']))
    
    plt.tight_layout()
    return fig


# ============================================================================
# FIGURE 2: System Architecture Diagram
# ============================================================================
def plot_system_architecture():
    """Draw Action-First pipeline architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    def draw_box(x, y, w, h, text, color, fontsize=10, bold=False):
        box = FancyBboxPatch((x, y), w, h, 
                             boxstyle="round,pad=0.05,rounding_size=0.2",
                             facecolor=color, edgecolor='white', linewidth=2,
                             alpha=0.95)
        ax.add_patch(box)
        weight = 'bold' if bold else 'normal'
        # Determine text color based on luminance
        if isinstance(color, str):
            is_light = color in ['#f8fafc', '#fef3c7', '#f0fdf4']
        else:
            # For numpy array colors, assume they are from Set3 colormap (light)
            is_light = True
        text_color = COLORS['text'] if is_light else 'white'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
                fontsize=fontsize, fontweight=weight, color=text_color)
    
    def draw_arrow(x1, y1, x2, y2, label='', color=COLORS['neutral']):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color=color, lw=2.5,
                                  connectionstyle='arc3,rad=0'))
        if label:
            mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
            ax.text(mid_x, mid_y + 0.15, label, fontsize=8, ha='center', 
                   style='italic', color=COLORS['neutral'])
    
    # Title
    ax.text(7, 9.5, 'Action-First Pipeline Architecture', 
            fontsize=18, fontweight='bold', ha='center', color=COLORS['text'])
    ax.text(7, 9.1, 'Real-time Workout Recognition & Rep Counting on Edge Device',
            fontsize=11, ha='center', color=COLORS['neutral'], style='italic')
    
    # Stage 0: Action Detection (left column)
    draw_box(0.5, 6.5, 3, 1.2, 'Stage 0\nAction Detection', COLORS['warning'], fontsize=11, bold=True)
    draw_box(0.5, 5.0, 3, 1.2, '6-axis IMU Input\n(100Hz, ax/ay/az/gx/gy/gz)', '#fef3c7', fontsize=9)
    draw_box(0.5, 3.5, 3, 1.2, 'Sliding Window\nFeature Extraction', '#fef3c7', fontsize=9)
    draw_box(0.5, 2.0, 3, 1.2, 'Action Classifier\n(8 classes)', '#fef3c7', fontsize=9)
    draw_arrow(2, 6.5, 2, 6.2)
    draw_arrow(2, 5.0, 2, 4.7)
    draw_arrow(2, 3.5, 2, 3.2)
    
    # Action switcher (center)
    draw_box(4.5, 4.5, 2.5, 2.5, 'Action\nRouter', COLORS['primary'], fontsize=12, bold=True)
    
    # Arrow from Stage 0 to Router
    ax.annotate('', xy=(4.5, 5.5), xytext=(3.5, 5.0),
               arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=3,
                              connectionstyle='arc3,rad=0.1'))
    ax.text(3.8, 5.8, 'detected\naction', fontsize=8, ha='center', color=COLORS['primary'])
    
    # Stage 1: Per-Action Rep Segmentation (right side, stacked vertically)
    draw_box(8, 7.5, 5.5, 1.5, 'Stage 1: Per-Action Rep Segmentation', COLORS['secondary'], fontsize=12, bold=True)
    
    # 8 action-specific models
    actions = ['bench_press', 'biceps_curl', 'rdl', 'shoulder_press', 
               'squat', 'triceps_curl', 'weighted_crunch', 'db_row']
    colors_act = plt.cm.Set3(np.linspace(0, 1, 8))
    
    for i, act in enumerate(actions):
        row = i // 4
        col = i % 4
        x = 8 + col * 1.35
        y = 6.0 - row * 1.0
        draw_box(x, y, 1.2, 0.8, act.replace('_', '\n'), colors_act[i], fontsize=7)
    
    # Router arrows to models
    for i in range(8):
        row = i // 4
        col = i % 4
        x_target = 8 + col * 1.35 + 0.6
        y_target = 6.0 - row * 1.0 + 0.8
        ax.annotate('', xy=(x_target, y_target), xytext=(7.0, 5.75),
                   arrowprops=dict(arrowstyle='->', color=COLORS['neutral'], lw=0.8,
                                  connectionstyle=f'arc3,rad={0.1*(i-3.5)}', alpha=0.4))
    
    # Stage 1 details box
    draw_box(8, 3.0, 5.5, 1.5, '', '#f0fdf4', fontsize=9)
    ax.text(10.75, 4.0, 'Per-Action Causal RF', fontsize=10, fontweight='bold', ha='center', color=COLORS['secondary'])
    ax.text(10.75, 3.55, '• 100 trees, depth 15\n• Trailing window (1.0s)\n• ~200KB per model', 
            fontsize=8, ha='center', color=COLORS['text'])
    
    # Output
    draw_box(8, 1.0, 5.5, 1.2, 'Output: Rep Count + Phase Labels\n(concentric/eccentric boundaries)', 
             COLORS['accent'], fontsize=10, bold=True)
    
    # Arrow from models to output
    ax.annotate('', xy=(10.75, 2.2), xytext=(10.75, 3.0),
               arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=2.5))
    
    # Key metrics box (bottom left)
    draw_box(0.5, 0.5, 6.5, 1.2, '', '#f8fafc', fontsize=9)
    ax.text(0.7, 1.3, 'Key Metrics:', fontsize=9, fontweight='bold', color=COLORS['text'])
    ax.text(0.7, 0.9, '• Rep F1 = 0.850  |  IoU-F1@50 = 0.706  |  Exact Count = 65.9%', 
            fontsize=8, color=COLORS['text'])
    ax.text(0.7, 0.6, '• Model Size: ~1.6MB total (8 models)  |  Latency: 1.0s causal window',
            fontsize=8, color=COLORS['text'])
    
    plt.tight_layout()
    return fig


# ============================================================================
# FIGURE 3: Feature Importance (Global + Per-Action)
# ============================================================================
def plot_feature_importance():
    """Plot global feature importance aggregated across all actions."""
    # Load data
    data_path = ROOT / "artifacts" / "feature_analysis" / "velocity_feature_importance.json"
    with open(data_path) as f:
        data = json.load(f)
    
    # Aggregate top features across all actions (average importance)
    feature_scores = {}
    for action, vals in data.items():
        for feat, score in vals.get("top_20", []):
            feature_scores[feat] = feature_scores.get(feat, 0) + score
    
    # Normalize by number of actions
    n_actions = len(data)
    for feat in feature_scores:
        feature_scores[feat] /= n_actions
    
    # Sort and take top 15
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:15]
    names = [f[0] for f in sorted_features]
    scores = [f[1] for f in sorted_features]
    
    # Categorize features by type for coloring
    def get_color(name):
        if 'vel_' in name:
            return COLORS['accent']  # Velocity features (red)
        elif any(c in name for c in ['ax', 'ay', 'az']):
            return COLORS['primary']  # Accelerometer (blue)
        elif any(c in name for c in ['gx', 'gy', 'gz']):
            return COLORS['secondary']  # Gyroscope (green)
        elif 'mag' in name:
            return COLORS['warning']  # Magnitude (orange)
        else:
            return COLORS['neutral']
    
    bar_colors = [get_color(n) for n in names]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, scores, color=bar_colors, edgecolor='white', linewidth=1.5, alpha=0.9)
    
    # Add score labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(score + 0.002, bar.get_y() + bar.get_height()/2, 
                f'{score:.4f}', va='center', fontsize=9, color=COLORS['text'])
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Average Feature Importance (MDI)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Action RF: Top 15 Features (Aggregated across 8 actions)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, max(scores) * 1.2)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['primary'], label='Accelerometer (ax/ay/az)'),
        mpatches.Patch(facecolor=COLORS['secondary'], label='Gyroscope (gx/gy/gz)'),
        mpatches.Patch(facecolor=COLORS['accent'], label='Velocity features'),
        mpatches.Patch(facecolor=COLORS['warning'], label='Magnitude'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    return fig


# ============================================================================
# FIGURE 4: Multi-dimensional Model Comparison (Why RF?)
# ============================================================================
def plot_model_comparison_radar():
    """Radar chart comparing all methods across 5 key dimensions."""
    
    # Data: [Rep F1, IoU-F1@50, Causal, Deployable, Stability]
    # Normalized to 0-1 scale for fair comparison
    # Causal/Deployable/Stability are binary (1 or 0) or approximated
    methods = {
        'Per-Action RF':     [0.850, 0.706, 1.0, 1.0, 1.0],
        'BiLSTM (Tuned)':    [0.831, 0.682, 0.0, 0.0, 0.6],
        'BiLSTM (Basic)':    [0.758, 0.549, 0.0, 0.0, 0.2],
        'Causal RF (Global)':[0.778, 0.561, 1.0, 1.0, 1.0],
        'Sliding-window RF': [0.768, 0.577, 0.0, 0.0, 1.0],
        'Peak Detection':    [0.755, 0.400, 1.0, 1.0, 0.5],
        'XGBoost':           [0.726, 0.538, 1.0, 0.3, 1.0],
        'CatBoost':          [0.720, 0.520, 1.0, 0.3, 1.0],
        '1D CNN':            [0.698, 0.464, 1.0, 0.3, 0.2],
    }
    
    categories = ['Rep F1', 'IoU-F1@50', 'Causal', 'Deployable', 'Stability']
    N = len(categories)
    
    # Compute angle for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the loop
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # Colors for each method
    method_colors = {
        'Per-Action RF':      '#2563eb',  # Blue (primary)
        'BiLSTM (Tuned)':     '#dc2626',  # Red
        'Causal RF (Global)': '#059669',  # Green
        'Peak Detection':     '#d97706',  # Orange
        'XGBoost':            '#64748b',  # Slate
        '1D CNN':             '#8b5cf6',  # Purple
    }
    
    # Plot each method
    for method, values in methods.items():
        if method not in method_colors:
            continue  # Skip methods not in color map
        values_plot = values + values[:1]  # Complete the loop
        color = method_colors[method]
        linewidth = 3 if 'Per-Action RF' in method else 1.5
        alpha = 0.9 if 'Per-Action RF' in method else 0.3
        zorder = 10 if 'Per-Action RF' in method else 1
        
        ax.plot(angles, values_plot, 'o-', linewidth=linewidth, label=method, 
                color=color, alpha=alpha, zorder=zorder)
        if 'Per-Action RF' in method:
            ax.fill(angles, values_plot, alpha=0.15, color=color, zorder=zorder)
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    
    # Set y-axis limits and labels
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9, color='gray')
    ax.grid(True, alpha=0.3)
    
    # Title
    ax.set_title('Model Comparison: Why Per-Action RF Wins\n(5 Dimensions, Normalized Scores)', 
                 fontsize=14, fontweight='bold', pad=30, y=1.08)
    
    # Legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=10, 
              framealpha=0.9, edgecolor='gray')
    
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Generating Figures for Presentation")
    print(f"Output directory: {FIG_DIR}")
    print("=" * 60)
    
    print("\n[1/4] Window Size Optimization Curve...")
    fig1 = plot_window_size_optimization()
    save_fig(fig1, "fig01_window_size_optimization")
    plt.close(fig1)
    
    print("\n[2/4] System Architecture Diagram...")
    fig2 = plot_system_architecture()
    save_fig(fig2, "fig02_system_architecture")
    plt.close(fig2)
    
    print("\n[3/4] Feature Importance Plot...")
    fig3 = plot_feature_importance()
    save_fig(fig3, "fig03_feature_importance")
    plt.close(fig3)
    
    print("\n[4/4] Model Comparison Radar Chart...")
    fig4 = plot_model_comparison_radar()
    save_fig(fig4, "fig04_model_comparison_radar")
    plt.close(fig4)
    
    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print(f"Location: {FIG_DIR}")
    print("=" * 60)
