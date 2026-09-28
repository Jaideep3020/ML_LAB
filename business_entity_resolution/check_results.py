import pandas as pd
import sys

def main():
    matching_df = pd.read_csv('output/matching_results.tsv', sep='\t', keep_default_na=False)
    candidate_df = pd.read_csv('output/candidate_pairs.tsv', sep='\t', keep_default_na=False)
    test_s1 = pd.read_csv('../work/full/data/dataset/test/test_source1.tsv', sep='\t', keep_default_na=False)

    print("Checking subsets...")
    subset_violations = 0
    for i, row in matching_df.iterrows():
        matches = str(row['matched_entity_ids']).split(',') if row['matched_entity_ids'] else []
        matches = [m for m in matches if m.strip()]
        
        cand_row = candidate_df.loc[candidate_df['source1_entity_id'] == row['source1_entity_id']]
        if not cand_row.empty:
            candidates = str(cand_row.iloc[0]['candidate_entity_ids']).split(',')
            candidates = [c for c in candidates if c.strip()]
            for match in matches:
                if match not in candidates:
                    print(f"Violation: {row['source1_entity_id']} matched {match} which is not in candidates.")
                    subset_violations += 1
    print(f"Subset violations: {subset_violations}")

    print("Checking singletons...")
    singleton_errors = 0
    for i, row in matching_df.iterrows():
        val = row['matched_entity_ids']
        if not val:
            continue
        if val in ('None', 'nan', ' ') or not val.strip():
            if val != '':
                print(f"Singleton error on {row['source1_entity_id']}: {repr(val)}")
                singleton_errors += 1
    print(f"Singleton errors: {singleton_errors}")

    print("Checking France random rows...")
    france_qids = test_s1[test_s1['country'] == 'France']['entity_id']
    france_matches = matching_df[matching_df['source1_entity_id'].isin(france_qids)]
    non_empty_france = france_matches[france_matches['matched_entity_ids'] != '']
    
    if len(non_empty_france) == 0:
        print("No matches for France to spot check.")
    else:
        sample = non_empty_france.sample(min(20, len(non_empty_france)))
        for i, row in sample.iterrows():
            print(f"{row['source1_entity_id']} -> {row['matched_entity_ids']}")

if __name__ == '__main__':
    main()
