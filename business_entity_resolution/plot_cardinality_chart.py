"""
Match Cardinality Distribution: Ground Truth vs Predicted by Country
Replicates the professional bar chart style with annotations.
"""
import pandas as pd
import numpy as np
import duckdb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

run_dir    = Path(r"C:\Users\jaide\Documents\Codex\2026-09-25\the\work\full")
db_path    = run_dir / 'cache' / 'data.duckdb'
truth_path = run_dir / 'data' / 'dataset' / 'train' / 'train_ground_truth.tsv'
desktop_tsv = r"C:\Users\jaide\OneDrive\Desktop\matching_results_neural.tsv"

con = duckdb.connect()
con.execute(f"ATTACH '{db_path}' AS db")

# ── Ground Truth ──────────────────────────────────────────────
truth_df = pd.read_csv(truth_path, sep='\t', dtype=str).fillna('')
truth_df['match_count'] = truth_df['matched_entity_ids'].apply(
    lambda x: len(x.split(',')) if x.strip() else 0
)
# Ground truth is ONLY US+India — combine them as one "GT (Train)" series
truth_df['cardinality_bin'] = truth_df['match_count'].apply(
    lambda x: 'k>5' if x > 5 else f'k={x}'
)

# ── Predictions ───────────────────────────────────────────────
preds_df = pd.read_csv(desktop_tsv, sep='\t', dtype=str).fillna('')
preds_df['match_count'] = preds_df['matched_entity_ids'].apply(
    lambda x: len(x.split(',')) if x.strip() else 0
)
preds_df['cardinality_bin'] = preds_df['match_count'].apply(
    lambda x: 'k>5' if x > 5 else f'k={x}'
)

# Get country for test_s1
test_s1 = con.execute("SELECT entity_id, country FROM db.test_s1").df()
test_s1.columns = ['source1_entity_id', 'country']
preds_merged = preds_df.merge(test_s1, on='source1_entity_id', how='left')

# ── Compute percentage distributions ──────────────────────────
bins_order = ['k=0', 'k=1', 'k=2', 'k=3', 'k=4', 'k=5', 'k>5']
bin_labels  = ['k=0\n(Singletons)', 'k=1', 'k=2', 'k=3', 'k=4', 'k=5', 'k>5\n(High Card.)']

def pct_dist(df, bin_col='cardinality_bin'):
    counts = df[bin_col].value_counts()
    total = len(df)
    return [(counts.get(b, 0) / total) * 100 for b in bins_order]

gt_pct     = pct_dist(truth_df)
india_pct  = pct_dist(preds_merged[preds_merged['country'] == 'India'])
us_pct     = pct_dist(preds_merged[preds_merged['country'] == 'US'])
france_pct = pct_dist(preds_merged[preds_merged['country'] == 'France'])

print("Distribution check:")
print(f"  GT:     {[f'{v:.1f}' for v in gt_pct]}")
print(f"  India:  {[f'{v:.1f}' for v in india_pct]}")
print(f"  US:     {[f'{v:.1f}' for v in us_pct]}")
print(f"  France: {[f'{v:.1f}' for v in france_pct]}")

# ── Plot ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor('#1e1e2e')
ax.set_facecolor('#1e1e2e')

n = len(bins_order)
x = np.arange(n)
w = 0.19  # bar width

colors = {
    'gt':     '#1a1a2e',   # very dark navy
    'india':  '#4361ee',   # blue
    'us':     '#2ecc71',   # green
    'france': '#e74c3c',   # red
}
edge = '#cccccc'

b1 = ax.bar(x - 1.5*w, gt_pct,     w, label='Ground Truth (Train)', color='#2c2c54',  edgecolor=edge, linewidth=0.5)
b2 = ax.bar(x - 0.5*w, india_pct,  w, label='Predicted: India',     color=colors['india'],  edgecolor=edge, linewidth=0.5)
b3 = ax.bar(x + 0.5*w, us_pct,     w, label='Predicted: US',        color=colors['us'],     edgecolor=edge, linewidth=0.5)
b4 = ax.bar(x + 1.5*w, france_pct, w, label='Predicted: France (Over-predicting)', color=colors['france'], edgecolor=edge, linewidth=0.5)

# ── Annotations ───────────────────────────────────────────────
# Annotation 1: k=1 under-prediction
k1_idx = bins_order.index('k=1')
# Compute actual k=1 inflation ratio
k1_ratio = india_pct[k1_idx] / gt_pct[k1_idx] if gt_pct[k1_idx] > 0 else 0
ax.annotate(
    f'India/US under-predict (k=1 inflated {k1_ratio:.1f}x)\nRoot cause: Threshold too high for 2nd/3rd matches',
    xy=(x[k1_idx] - 0.5*w, india_pct[k1_idx] + 0.5),
    xytext=(x[k1_idx] - 0.8, india_pct[k1_idx] + 7),
    fontsize=9, color='white',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#4361ee', alpha=0.85, edgecolor='white'),
    arrowprops=dict(arrowstyle='->', color='white', lw=1.5)
)

# Annotation 2: k>5 ALL countries under-predict vs GT (corrected from 'over-predicts')
k5p_idx = bins_order.index('k>5')
deficit_pct = gt_pct[k5p_idx] - france_pct[k5p_idx]
ax.annotate(
    f'ALL countries under-predict k>5\n({france_pct[k5p_idx]:.2f}% vs {gt_pct[k5p_idx]:.2f}% GT, -{deficit_pct:.2f}%)\nRoot cause: BM25 top-k=10 ceiling (fix: top-k=20)',
    xy=(x[k5p_idx] + 1.5*w, france_pct[k5p_idx] + 0.5),
    xytext=(x[k5p_idx] - 0.5, france_pct[k5p_idx] + 9),
    fontsize=9, color='white',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#c0392b', alpha=0.90, edgecolor='white'),
    arrowprops=dict(arrowstyle='->', color='white', lw=1.5)
)

# Annotation 3: k=4 deficit — secondary threshold effect
k4_idx = bins_order.index('k=4')
ax.annotate(
    f'k=4 deflated across all countries\nGT: {gt_pct[k4_idx]:.1f}% | India: {india_pct[k4_idx]:.1f}%',
    xy=(x[k4_idx] - 0.5*w, india_pct[k4_idx] - 0.5),
    xytext=(x[k4_idx] - 1.2, india_pct[k4_idx] - 6),
    fontsize=8, color='#aaaaaa',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#2c2c54', alpha=0.75, edgecolor='#555555'),
    arrowprops=dict(arrowstyle='->', color='#aaaaaa', lw=1.2)
)

# ── Styling ───────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(bin_labels, color='white', fontsize=10)
ax.set_ylabel('Percentage of Queries (%)', color='white', fontsize=11)
ax.set_title('Match Cardinality Distribution: Ground Truth vs Predicted by Country',
             color='white', fontsize=13, fontweight='bold', pad=15)
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('#555555')
ax.yaxis.set_tick_params(labelcolor='white')
ax.set_ylim(0, max(max(gt_pct), max(france_pct)) + 14)

# Grid
ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='white')
ax.set_axisbelow(True)

# Legend
legend = ax.legend(facecolor='#2c2c54', edgecolor='#555555', labelcolor='white',
                   fontsize=9, loc='upper right')

plt.tight_layout()
plt.savefig('cardinality_distribution_chart.png', dpi=180, bbox_inches='tight',
            facecolor='#1e1e2e')
print("Saved cardinality_distribution_chart.png!")
plt.show()
