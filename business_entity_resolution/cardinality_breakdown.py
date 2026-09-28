"""
Cardinality & Distribution Breakdown
Ground Truth (Training) vs Predicted (Test) by Country
"""
import pandas as pd
import numpy as np
import duckdb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

sns.set_theme(style="darkgrid")
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

run_dir = Path(r"C:\Users\jaide\Documents\Codex\2026-09-25\the\work\full")
db_path   = run_dir / 'cache' / 'data.duckdb'
truth_path = run_dir / 'data' / 'dataset' / 'train' / 'train_ground_truth.tsv'
desktop_tsv = r"C:\Users\jaide\OneDrive\Desktop\matching_results_neural.tsv"

print("="*60)
print("CARDINALITY & DISTRIBUTION BREAKDOWN")
print("Ground Truth (Train) vs Predicted (Test) — By Country")
print("="*60)

con = duckdb.connect()
con.execute(f"ATTACH '{db_path}' AS db")

# ────────────────────────────────────────────────────────────
# PART A: Ground Truth Statistics from Training Set
# ────────────────────────────────────────────────────────────
print("\n[A] Computing Ground Truth Cardinality from Training Set...")

truth_df = pd.read_csv(truth_path, sep='\t', dtype=str).fillna('')
truth_df['match_count'] = truth_df['matched_entity_ids'].apply(
    lambda x: len(x.split(',')) if x.strip() else 0
)

# Get country for each train_s1 entity
train_s1_country = con.execute(
    "SELECT entity_id, country FROM db.train_s1"
).df()
train_s1_country.columns = ['source1_entity_id', 'country']

truth_merged = truth_df.merge(train_s1_country, on='source1_entity_id', how='left')

print("\n--- Ground Truth Cardinality by Country (Training Set) ---")
gt_by_country = truth_merged.groupby('country').agg(
    total_entities=('source1_entity_id', 'count'),
    singletons=('match_count', lambda x: (x == 0).sum()),
    avg_matches=('match_count', 'mean'),
    max_matches=('match_count', 'max'),
    pct_multi=('match_count', lambda x: (x > 1).mean() * 100)
).reset_index()
gt_by_country['singleton_rate_pct'] = gt_by_country['singletons'] / gt_by_country['total_entities'] * 100
print(gt_by_country.to_string(index=False))

# Distribution of match counts per country (training)
gt_dist = truth_merged.groupby(['country', 'match_count']).size().reset_index(name='count')

# ────────────────────────────────────────────────────────────
# PART B: Predicted Statistics from Test Submission
# ────────────────────────────────────────────────────────────
print("\n[B] Computing Predicted Cardinality from Test Submission...")

preds_df = pd.read_csv(desktop_tsv, sep='\t', dtype=str).fillna('')
preds_df['match_count'] = preds_df['matched_entity_ids'].apply(
    lambda x: len(x.split(',')) if x.strip() else 0
)

# Get country for each test_s1 entity
test_s1_country = con.execute(
    "SELECT entity_id, country FROM db.test_s1"
).df()
test_s1_country.columns = ['source1_entity_id', 'country']

preds_merged = preds_df.merge(test_s1_country, on='source1_entity_id', how='left')

print("\n--- Predicted Cardinality by Country (Test Set Submission) ---")
pred_by_country = preds_merged.groupby('country').agg(
    total_entities=('source1_entity_id', 'count'),
    singletons=('match_count', lambda x: (x == 0).sum()),
    avg_matches=('match_count', 'mean'),
    max_matches=('match_count', 'max'),
    pct_multi=('match_count', lambda x: (x > 1).mean() * 100)
).reset_index()
pred_by_country['singleton_rate_pct'] = pred_by_country['singletons'] / pred_by_country['total_entities'] * 100
print(pred_by_country.to_string(index=False))

# Distribution of match counts per country (test)
pred_dist = preds_merged.groupby(['country', 'match_count']).size().reset_index(name='count')

# ────────────────────────────────────────────────────────────
# PART C: Gap Analysis
# ────────────────────────────────────────────────────────────
print("\n[C] Gap Analysis: Predicted vs Ground Truth Drift...")
# Only compare countries present in BOTH
common_countries = set(gt_by_country['country']) & set(pred_by_country['country'])
print(f"\nCommon countries (train & test): {sorted(common_countries)}")

gt_common = gt_by_country[gt_by_country['country'].isin(common_countries)].set_index('country')
pred_common = pred_by_country[pred_by_country['country'].isin(common_countries)].set_index('country')

print("\n--- Avg Matches Drift (Predicted - Ground Truth) ---")
drift = pred_common[['avg_matches','singleton_rate_pct']] - gt_common[['avg_matches','singleton_rate_pct']]
drift.columns = ['avg_matches_drift', 'singleton_rate_drift_pct']
print(drift)
print("\n  Positive avg_matches_drift → Model is OVER-predicting matches (false positives)")
print("  Negative avg_matches_drift → Model is UNDER-predicting matches (false negatives)")

# ────────────────────────────────────────────────────────────
# PART D: Plots
# ────────────────────────────────────────────────────────────
print("\n[D] Generating Cardinality & Distribution Graphs...")

fig = plt.figure(figsize=(22, 20))
gs = gridspec.GridSpec(3, 2, hspace=0.5, wspace=0.35)

# ── Plot 1: Avg Matches per Entity by Country (GT vs Pred)
ax1 = fig.add_subplot(gs[0, 0])
countries_in_gt = gt_by_country['country'].tolist()
x = np.arange(len(countries_in_gt))
w = 0.35
bars1 = ax1.bar(x - w/2, gt_by_country['avg_matches'], w, label='Ground Truth (Train)', color='#6C5CE7', alpha=0.85)
gt_idx = {c: i for i, c in enumerate(countries_in_gt)}
pred_avg = [pred_by_country[pred_by_country['country']==c]['avg_matches'].values[0]
            if c in pred_by_country['country'].values else 0 for c in countries_in_gt]
bars2 = ax1.bar(x + w/2, pred_avg, w, label='Predicted (Test)', color='#00B894', alpha=0.85)
ax1.set_title("Avg Matches per Entity by Country\n(GT vs Predicted)", fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(countries_in_gt)
ax1.set_ylabel("Avg Match Count")
ax1.legend()
for bar in bars1: ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.03, f'{bar.get_height():.2f}', ha='center', fontsize=8)
for bar in bars2: ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.03, f'{bar.get_height():.2f}', ha='center', fontsize=8)

# ── Plot 2: Singleton Rate by Country (GT vs Pred)
ax2 = fig.add_subplot(gs[0, 1])
pred_srate = [pred_by_country[pred_by_country['country']==c]['singleton_rate_pct'].values[0]
              if c in pred_by_country['country'].values else 0 for c in countries_in_gt]
ax2.bar(x - w/2, gt_by_country['singleton_rate_pct'], w, label='Ground Truth (Train)', color='#FD79A8', alpha=0.85)
ax2.bar(x + w/2, pred_srate, w, label='Predicted (Test)', color='#FDCB6E', alpha=0.85)
ax2.set_title("Singleton Rate (%) by Country\n(GT vs Predicted)", fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(countries_in_gt)
ax2.set_ylabel("Singleton Rate (%)")
ax2.legend()

# ── Plot 3: Match Count Distribution — Ground Truth per Country
ax3 = fig.add_subplot(gs[1, 0])
for country in gt_dist['country'].unique():
    sub = gt_dist[gt_dist['country'] == country]
    sub = sub[sub['match_count'] <= 8]
    ax3.plot(sub['match_count'], sub['count'], marker='o', label=country, linewidth=2)
ax3.set_title("Match Count Distribution\n(Ground Truth, Training Set)", fontweight='bold')
ax3.set_xlabel("Match Count per Source 1 Entity")
ax3.set_ylabel("Number of Entities")
ax3.legend()

# ── Plot 4: Match Count Distribution — Predicted per Country
ax4 = fig.add_subplot(gs[1, 1])
for country in pred_dist['country'].unique():
    sub = pred_dist[pred_dist['country'] == country]
    sub = sub[sub['match_count'] <= 8]
    ax4.plot(sub['match_count'], sub['count'], marker='s', label=country, linewidth=2)
ax4.set_title("Match Count Distribution\n(Predicted, Test Submission)", fontweight='bold')
ax4.set_xlabel("Match Count per Source 1 Entity")
ax4.set_ylabel("Number of Entities")
ax4.legend()

# ── Plot 5: Entity Count by Country (Train vs Test volumes)
ax5 = fig.add_subplot(gs[2, 0])
all_countries_gt = gt_by_country['country'].tolist()
all_countries_pred = pred_by_country['country'].tolist()
all_countries = sorted(set(all_countries_gt + all_countries_pred))
gt_totals   = [gt_by_country[gt_by_country['country']==c]['total_entities'].values[0] if c in all_countries_gt else 0 for c in all_countries]
pred_totals = [pred_by_country[pred_by_country['country']==c]['total_entities'].values[0] if c in all_countries_pred else 0 for c in all_countries]
x2 = np.arange(len(all_countries))
ax5.bar(x2 - w/2, gt_totals, w, label='Train Entities', color='#74B9FF', alpha=0.85)
ax5.bar(x2 + w/2, pred_totals, w, label='Test Entities', color='#55EFC4', alpha=0.85)
ax5.set_title("Total Entity Count by Country\n(Train vs Test Volume)", fontweight='bold')
ax5.set_xticks(x2)
ax5.set_xticklabels(all_countries)
ax5.set_ylabel("Entity Count")
ax5.legend()
for i, (g, p) in enumerate(zip(gt_totals, pred_totals)):
    ax5.text(i - w/2, g + 100, f'{g:,}', ha='center', fontsize=7, rotation=45)
    ax5.text(i + w/2, p + 100, f'{p:,}', ha='center', fontsize=7, rotation=45)

# ── Plot 6: Avg Matches Drift heatmap
ax6 = fig.add_subplot(gs[2, 1])
drift_reset = drift.reset_index()
heatmap_data = drift_reset.set_index('country')[['avg_matches_drift', 'singleton_rate_drift_pct']]
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn_r', center=0, ax=ax6, linewidths=0.5)
ax6.set_title("Prediction Drift Heatmap\n(Predicted − Ground Truth)\nPositive = Over-predicting", fontweight='bold')

plt.suptitle("Cardinality & Distribution Breakdown: Ground Truth vs Predicted by Country\nLeaderboard F0.5: 0.879", 
             fontsize=14, fontweight='bold', y=1.01)
plt.savefig("cardinality_country_breakdown.png", bbox_inches='tight')
print("  Saved cardinality_country_breakdown.png!")

# ────────────────────────────────────────────────────────────
# PART E: Print final summary
# ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("ORCHESTRATOR FINDINGS SUMMARY")
print("="*60)
print("\n=== GROUND TRUTH (Training) ===")
print(gt_by_country[['country','total_entities','singleton_rate_pct','avg_matches','max_matches']].to_string(index=False))
print("\n=== PREDICTED (Test Submission) ===")
print(pred_by_country[['country','total_entities','singleton_rate_pct','avg_matches','max_matches']].to_string(index=False))
print("\n=== DRIFT (Predicted - Ground Truth) ===")
print(drift.to_string())

# France specific check
france_pred = pred_by_country[pred_by_country['country'].str.upper().str.contains('FR|FRANCE', na=False)]
if len(france_pred):
    print(f"\n🔴 FRANCE (OOD Country) Predicted Stats:")
    print(france_pred[['country','total_entities','singleton_rate_pct','avg_matches']].to_string(index=False))
else:
    print("\n🔴 France not found in predictions — check if it was handled.")
    
print("\n✅ Analysis Complete!")
print("  → cardinality_country_breakdown.png (6-panel graph)")
