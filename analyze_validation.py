import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def compute_f05(precision, recall):
    if precision == 0 and recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)

def main():
    parser = argparse.ArgumentParser(description="Analyze predictions against ground truth")
    parser.add_argument('--predictions', required=True, help="Path to your TSV prediction file")
    parser.add_argument('--truth', required=True, help="Path to train_ground_truth.tsv")
    args = parser.parse_args()

    print("Loading predictions...")
    pred_df = pd.read_csv(args.predictions, sep='\t', dtype=str).fillna('')
    print("Loading ground truth...")
    truth_df = pd.read_csv(args.truth, sep='\t', dtype=str).fillna('')

    # Merge
    merged = pd.merge(truth_df, pred_df, on='source1_entity_id', how='inner', suffixes=('_truth', '_pred'))
    
    if len(merged) == 0:
        print("Error: No overlapping entity IDs found. You might be trying to evaluate Test Set predictions against the Train Set ground truth!")
        return

    print(f"Analyzing {len(merged)} overlapping entities...")

    total_precision = 0.0
    total_recall = 0.0
    false_positives_list = []
    false_negatives_list = []

    for _, row in merged.iterrows():
        s1_id = row['source1_entity_id']
        t_str = row['matched_entity_ids_truth'].strip()
        p_str = row['matched_entity_ids_pred'].strip()
        
        t_set = set(t_str.split(',')) if t_str else set()
        p_set = set(p_str.split(',')) if p_str else set()
        
        # True Positives
        tp = t_set.intersection(p_set)
        fp = p_set - t_set
        fn = t_set - p_set
        
        # If both are empty (Singleton correct)
        if len(t_set) == 0 and len(p_set) == 0:
            precision = 1.0
            recall = 1.0
        else:
            precision = len(tp) / len(p_set) if len(p_set) > 0 else 0.0
            recall = len(tp) / len(t_set) if len(t_set) > 0 else 0.0
            
        total_precision += precision
        total_recall += recall
        
        if len(fp) > 0:
            false_positives_list.append({'source1': s1_id, 'false_claims': ','.join(fp), 'missed_truth': ','.join(fn)})
            
        if len(fn) > 0:
            false_negatives_list.append({'source1': s1_id, 'missed_truth': ','.join(fn)})

    macro_precision = total_precision / len(merged)
    macro_recall = total_recall / len(merged)
    macro_f05 = compute_f05(macro_precision, macro_recall)

    print("\n" + "="*40)
    print("🏆 FINAL METRICS 🏆")
    print("="*40)
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall:    {macro_recall:.4f}")
    print(f"Macro F0.5:      {macro_f05:.4f}")
    print(f"\nTotal False Positives (Bad Merges): {len(false_positives_list)}")
    print(f"Total False Negatives (Missed):   {len(false_negatives_list)}")
    
    # Save False Positives for inspection
    if len(false_positives_list) > 0:
        fp_df = pd.DataFrame(false_positives_list)
        fp_df.to_csv("false_positives_analysis.csv", index=False)
        print("Saved false positives list to 'false_positives_analysis.csv' for you to inspect!")

    # Generate Graphs
    plt.figure(figsize=(10, 5))
    metrics = ['Precision', 'Recall', 'F0.5']
    values = [macro_precision, macro_recall, macro_f05]
    
    sns.barplot(x=metrics, y=values, palette="viridis")
    plt.title("Pipeline Performance Metrics")
    plt.ylim(0.0, 1.0)
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold')
        
    plt.savefig('metrics_chart.png')
    print("Saved bar chart to 'metrics_chart.png'")

if __name__ == '__main__':
    main()
