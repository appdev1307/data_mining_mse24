# ------------------------------------------------------
# Optimized GSE2361 FP-Growth Analysis with Association Rules
# ------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
from scipy.sparse import csr_matrix

print("🔄 Step 1: Loading gene expression data from GSE2361...")

# === LOAD DATA ===
# Optimization: Specify dtypes for PROBE_ID as string, skip metadata rows
file_path = "/GSE2361_series_matrix.txt"  # Ensure file is unzipped
try:
    df_raw = pd.read_csv(file_path, sep="\t", comment='!', skiprows=0, dtype={0: str}, low_memory=False)
except Exception as e:
    print(f"⚠️ Error loading file: {e}")
    raise

df_raw.columns = df_raw.columns.str.strip()
df_raw = df_raw.rename(columns={df_raw.columns[0]: "PROBE_ID"})
df_raw = df_raw.iloc[:5000, :10]  # Test with 5,000 genes, 10 samples
df_raw = df_raw.dropna()

print("✅ Loaded raw matrix with shape:", df_raw.shape)

# === CLEAN DATA ===
print("🧹 Step 2: Cleaning and formatting data...")
# Set index and remove duplicates
df_raw = df_raw.set_index("PROBE_ID").drop_duplicates()

# Diagnostic: Check for non-numeric columns
non_numeric_cols = df_raw.select_dtypes(exclude=['float', 'int']).columns
if len(non_numeric_cols) > 0:
    print(f"⚠️ Found {len(non_numeric_cols)} non-numeric columns: {non_numeric_cols.tolist()}")

# Convert all columns to numeric, coercing errors to NaN
for col in df_raw.columns:
    df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

# Drop rows/columns with any NaNs after conversion
df_raw = df_raw.dropna()
# Convert to float32 for memory efficiency
df_raw = df_raw.astype('float32')

# Diagnostic: Verify all values are numeric
if not df_raw.select_dtypes(include=['float32']).shape[1] == df_raw.shape[1]:
    print("⚠️ Warning: Some columns are still non-numeric after conversion!")
    raise ValueError("Non-numeric data remains in DataFrame")

print("✅ Cleaned matrix shape:", df_raw.shape)

# === BINARIZE ===
print("🔎 Step 3: Binarizing expression (threshold > 6.0)...")
threshold = 6.5
# Vectorized binarization
df_bin = df_raw > threshold

print(f"✅ Binarization complete. Sample shape: {df_bin.shape}")

# === TRANSACTIONS ===
print("🛍️ Step 4: Converting expression matrix to transactions...")
# List comprehension for transactions
transactions = [df_bin.index[df_bin[col]].tolist() for col in df_bin.columns]
# Diagnostic: Print average transaction size
print("Average genes per transaction:", np.mean([len(t) for t in transactions]))

print(f"✅ Created {len(transactions)} transactions")

# === ENCODE ===
print("🧠 Step 5: Encoding transactions using TransactionEncoder...")
te = TransactionEncoder()
# Optimization: Use sparse matrix
te_ary = te.fit(transactions).transform(transactions, sparse=True)
df_encoded = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)

print("✅ Transaction matrix shape:", df_encoded.shape)

# === FP-GROWTH ===
print("🌲 Step 6: Running FP-Growth algorithm...")
# Optimization: High min_support for 37 samples, limit itemset size
min_support = 0.65  # ~24/37 samples
frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True, max_len=3)

print(f"✅ Found {len(frequent_itemsets)} frequent itemsets (min_support={min_support})")

# === RULES ===
print("🔗 Step 7: Generating association rules...")
# Optimization: High min_confidence for fewer rules
min_confidence = 0.85
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
rules_sorted = rules.sort_values(by='lift', ascending=False)

print(f"✅ Found {len(rules_sorted)} association rules (min_confidence={min_confidence})")

# === VISUALIZATION ===
print("📊 Step 8: Plotting top rules...")
# Optimization: Plot top 50 rules
top_rules = rules_sorted.head(50)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=top_rules, x='support', y='confidence', size='lift', hue='lift',
                palette='viridis', sizes=(100, 1000))
plt.title("Top 50 Association Rules from GSE2361 (FP-Growth)")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.grid(True)
plt.tight_layout()
plt.show()

# === SAVE OUTPUTS ===
print("💾 Step 9: Saving results to CSV files...")
frequent_itemsets.to_csv("frequent_itemsets_gse2361.csv", index=False, compression=None)
rules_sorted.to_csv("association_rules_gse2361.csv", index=False, compression=None)

print("✅ All done! Results saved as:")
print("   → frequent_itemsets_gse2361.csv")
print("   → association_rules_gse2361.csv")
