# ------------------------------------------------------
# GSE2361 FP-Growth Analysis with Association Rules
# ------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

print("🔄 Step 1: Loading gene expression data from GSE2361...")

# === LOAD DATA ===
file_path = "GSE2361_series_matrix.txt"  # Make sure file is unzipped
df_raw = pd.read_csv(file_path, sep="\t", comment='!', skiprows=0)
df_raw.columns = df_raw.columns.str.strip()
df_raw = df_raw.rename(columns={df_raw.columns[0]: "PROBE_ID"})
df_raw = df_raw.dropna()

print("✅ Loaded raw matrix with shape:", df_raw.shape)

# === CLEAN DATA ===
print("🧹 Step 2: Cleaning and formatting data...")
df_raw.set_index("PROBE_ID", inplace=True)
df_raw = df_raw[~df_raw.index.duplicated(keep='first')]
df_raw = df_raw.apply(pd.to_numeric, errors='coerce')
df_raw = df_raw.dropna()

print("✅ Cleaned matrix shape:", df_raw.shape)

# === BINARIZE ===
print("🔎 Step 3: Binarizing expression (threshold > 6.0)...")
threshold = 6.5
df_bin = df_raw.applymap(lambda x: x > threshold)

print(f"✅ Binarization complete. Sample shape: {df_bin.shape}")

# === TRANSACTIONS ===
print("🛍️ Step 4: Converting expression matrix to transactions...")
transactions = []
for sample in df_bin.columns:
    expressed_genes = df_bin.index[df_bin[sample]].tolist()
    transactions.append(expressed_genes)

print(f"✅ Created {len(transactions)} transactions")

# === ENCODE ===
print("🧠 Step 5: Encoding transactions using TransactionEncoder...")
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

print("✅ Transaction matrix shape:", df_encoded.shape)

# === FP-GROWTH ===
print("🌲 Step 6: Running FP-Growth algorithm...")
min_support = 0.5
frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)

print(f"✅ Found {len(frequent_itemsets)} frequent itemsets (min_support={min_support})")

# === RULES ===
print("🔗 Step 7: Generating association rules...")
min_confidence = 0.7
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
rules_sorted = rules.sort_values(by='lift', ascending=False)

print(f"✅ Found {len(rules_sorted)} association rules (min_confidence={min_confidence})")

# === VISUALIZATION ===
print("📊 Step 8: Plotting top rules...")
plt.figure(figsize=(10, 6))
sns.scatterplot(data=rules_sorted, x='support', y='confidence', size='lift', hue='lift',
                palette='viridis', sizes=(100, 1000))
plt.title("Top Association Rules from GSE2361 (FP-Growth)")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.grid(True)
plt.tight_layout()
plt.show()

# === SAVE OUTPUTS ===
print("💾 Step 9: Saving results to CSV files...")
frequent_itemsets.to_csv("frequent_itemsets_gse2361.csv", index=False)
rules_sorted.to_csv("association_rules_gse2361.csv", index=False)

print("✅ All done! Results saved as:")
print("   → frequent_itemsets_gse2361.csv")
print("   → association_rules_gse2361.csv")
