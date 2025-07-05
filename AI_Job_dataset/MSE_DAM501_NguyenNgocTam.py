import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Load the dataset
data = pd.read_csv('ai_job_dataset.csv')

# Preprocess: Split 'required_skills' column into lists of skills (normalize case)
transactions = data['required_skills'].str.lower().str.split(',').apply(lambda x: [item.strip() for item in x]).tolist()

# Encode transactions into a binary matrix
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Step 1: Find frequent itemsets using FP-growth
min_support = 0.02  # 2% support
frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)

# Step 2: Filter for maximal frequent itemsets (for plotting)
def is_maximal(itemset, frequent_itemsets):
    """Check if an itemset is maximal (no superset is frequent)."""
    itemset_set = set(itemset)
    for other_itemset in frequent_itemsets['itemsets']:
        if set(other_itemset).issuperset(itemset_set) and other_itemset != itemset:
            return False
    return True

maximal_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(
    lambda x: is_maximal(x, frequent_itemsets)
)]

# Step 3: Generate association rules from all frequent itemsets
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)

# Step 4: Format and save maximal itemsets to CSV
maximal_itemsets_df = maximal_itemsets.copy()
maximal_itemsets_df['itemsets'] = maximal_itemsets_df['itemsets'].apply(lambda x: ', '.join(list(x)))
maximal_itemsets_df.to_csv('maximal_itemsets.csv', index=False)

# Step 5: Format and save association rules to CSV
if not rules.empty:
    rules_df = rules.copy()
    rules_df['antecedents'] = rules_df['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules_df['consequents'] = rules_df['consequents'].apply(lambda x: ', '.join(list(x)))
    rules_df.to_csv('association_rules.csv', index=False)
else:
    print("No association rules generated. Saving empty association_rules.csv.")
    pd.DataFrame(columns=['antecedents', 'consequents', 'support', 'confidence', 'lift']).to_csv('association_rules.csv', index=False)

# Step 6: Output results
print("Maximal Frequent Itemsets:")
for _, row in maximal_itemsets.iterrows():
    print(f"Itemset: {list(row['itemsets'])}, Support: {row['support']:.4f}")

print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Save results to text file
with open('frequent_patterns_and_rules.txt', 'w') as f:
    f.write("Maximal Frequent Itemsets:\n")
    for _, row in maximal_itemsets.iterrows():
        f.write(f"Itemset: {list(row['itemsets'])}, Support: {row['support']:.4f}\n")
    f.write("\nAssociation Rules:\n")
    f.write(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_string())

# Step 7: Plotting with Matplotlib
# Plot 1: Top 20 Maximal Frequent Itemsets by Support
top_20_itemsets = maximal_itemsets.sort_values(by='support', ascending=False).head(20)
plt.figure(figsize=(12, 6))
itemset_labels = [', '.join(list(itemset)) for itemset in top_20_itemsets['itemsets']]
plt.bar(itemset_labels, top_20_itemsets['support'], color='skyblue', edgecolor='black')
plt.xlabel('Itemsets')
plt.ylabel('Support')
plt.title('Top 20 Maximal Frequent Itemsets by Support')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Plot 2: Association Rules Confidence (if rules exist)
if not rules.empty:
    plt.figure(figsize=(12, 6))
    rule_labels = [f"{', '.join(list(rule['antecedents']))} → {', '.join(list(rule['consequents']))}" for _, rule in rules.iterrows()]
    plt.bar(rule_labels, rules['confidence'], color='lightgreen', edgecolor='black')
    plt.xlabel('Association Rules')
    plt.ylabel('Confidence')
    plt.title('Confidence of Association Rules')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print("No association rules to plot.")

# Step 8: Chart.js configurations for UI
chartjs_itemsets = {
    "type": "bar",
    "data": {
        "labels": itemset_labels,
        "datasets": [{
            "label": "Support",
            "data": top_20_itemsets['support'].tolist(),
            "backgroundColor": ["#1f77b4"] * len(itemset_labels),
            "borderColor": "#ffffff",
            "borderWidth": 1
        }]
    },
    "options": {
        "scales": {
            "y": {
                "beginAtZero": True,
                "title": { "display": True, "text": "Support" }
            },
            "x": {
                "title": { "display": True, "text": "Itemsets" }
            }
        },
        "plugins": {
            "legend": { "display": True },
            "title": { "display": True, "text": "Top 20 Maximal Frequent Itemsets by Support" }
        }
    }
}

chartjs_rules = {
    "type": "bar",
    "data": {
        "labels": rule_labels if not rules.empty else [],
        "datasets": [{
            "label": "Confidence",
            "data": rules['confidence'].tolist() if not rules.empty else [],
            "backgroundColor": ["#2ca02c"] * len(rules) if not rules.empty else [],
            "borderColor": "#ffffff",
            "borderWidth": 1
        }]
    },
    "options": {
        "scales": {
            "y": {
                "beginAtZero": True,
                "title": { "display": True, "text": "Confidence" }
            },
            "x": {
                "title": { "display": True, "text": "Association Rules" }
            }
        },
        "plugins": {
            "legend": { "display": True },
            "title": { "display": True, "text": "Confidence of Association Rules" }
        }
    }
}

print("\nChart.js Configuration for Top 20 Maximal Frequent Itemsets:")
print(chartjs_itemsets)
print("\nChart.js Configuration for Association Rules:")
print(chartjs_rules)