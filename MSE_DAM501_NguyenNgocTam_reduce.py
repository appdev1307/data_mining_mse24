# ------------------------------------------------------
# Optimized GSE2361 FP-Growth Analysis with Association Rules (200 x 37 Subset)
# ------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
from scipy.sparse import csr_matrix
import time

# Function to track time for each step
def print_time(step_name, start_time):
    elapsed = time.time() - start_time
    print(f"⏱️ {step_name} took {elapsed:.2f} seconds")

print("🔄 Step 1: Loading gene expression data from GSE2361...")
start_time = time.time()

# === LOAD DATA ===
file_path = "GSE2361_series_matrix.txt"  # Ensure file is unzipped
try:
    df_raw = pd.read_csv(file_path, sep="\t", comment='!', skiprows=0, dtype={0: str}, low_memory=False)
except Exception as e:
    print(f"⚠️ Error loading file: {e}")
    raise

df_raw.columns = df_raw.columns.str.strip()
df_raw = df_raw.rename(columns={df_raw.columns[0]: "PROBE_ID"})
df_raw = df_raw.dropna()

# Subset to 200 genes x 37 columns (PROBE_ID + 36 samples)
df_raw = df_raw.iloc[:400, :37]

print("✅ Loaded and subset raw matrix with shape:", df_raw.shape)
print_time("Loading and subsetting", start_time)

# === CLEAN DATA ===
print("🧹 Step 2: Cleaning and formatting data...")
start_time = time.time()

df_raw = df_raw.set_index("PROBE_ID").drop_duplicates()

# Diagnostic: Check for non-numeric columns
non_numeric_cols = df_raw.select_dtypes(exclude=['float', 'int']).columns
if len(non_numeric_cols) > 0:
    print(f"⚠️ Found {len(non_numeric_cols)} non-numeric columns: {non_numeric_cols.tolist()}")

# Convert all columns to numeric, coercing errors to NaN
for col in df_raw.columns:
    df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

# Drop rows/columns with any NaNs
df_raw = df_raw.dropna()
# Convert to float32 for memory efficiency
df_raw = df_raw.astype('float32')

# Verify all values are numeric
if not df_raw.select_dtypes(include=['float32']).shape[1] == df_raw.shape[1]:
    print("⚠️ Warning: Some columns are still non-numeric after conversion!")
    raise ValueError("Non-numeric data remains in DataFrame")

# Diagnostic: Print data range to guide threshold selection
print(f"Data range: {df_raw.min().min():.2f} to {df_raw.max().max():.2f}")
# Diagnostic: Print quantiles to guide threshold
quantiles = df_raw.quantile([0.5, 0.75, 0.9, 0.95]).transpose()
print("Quantiles (median, 75th, 90th, 95th):\n", quantiles)

print("✅ Cleaned matrix shape:", df_raw.shape)
print_time("Cleaning and formatting", start_time)

# === BINARIZE ===
print("🔎 Step 3: Binarizing expression (threshold > 100.0)...")
start_time = time.time()

# Higher threshold based on data range
threshold = 100.0
df_bin = df_raw > threshold

# Diagnostic: Percentage of genes above threshold
pct_above_threshold = df_bin.mean().mean() * 100
print(f"Percentage of genes above threshold: {pct_above_threshold:.2f}%")

print(f"✅ Binarization complete. Sample shape: {df_bin.shape}")
print_time("Binarization", start_time)

# === TRANSACTIONS ===
print("🛍️ Step 4: Converting expression matrix to transactions...")
start_time = time.time()

# List comprehension for transactions
transactions = [df_bin.index[df_bin[col]].tolist() for col in df_bin.columns]
# Diagnostic: Print transaction size statistics
avg_trans_size = np.mean([len(t) for t in transactions])
max_trans_size = np.max([len(t) for t in transactions])
print(f"Average genes per transaction: {avg_trans_size:.2f}")
print(f"Maximum genes per transaction: {max_trans_size}")

print(f"✅ Created {len(transactions)} transactions")
print_time("Transaction creation", start_time)

# === ENCODE ===
print("🧠 Step 5: Encoding transactions using TransactionEncoder...")
start_time = time.time()

te = TransactionEncoder()
# Optimization: Use sparse matrix
te_ary = te.fit(transactions).transform(transactions, sparse=True)
df_encoded = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)

print("✅ Transaction matrix shape:", df_encoded.shape)
print_time("Transaction encoding", start_time)

# === FP-GROWTH ===
print("🌲 Step 6: Running FP-Growth algorithm...")
start_time = time.time()

# Optimization: Adjust min_support for 36 samples
min_support = 0.7  # ~25/36 samples
frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True, max_len=2)

print(f"✅ Found {len(frequent_itemsets)} frequent itemsets (min_support={min_support})")
print_time("FP-Growth", start_time)

# === RULES ===
print("🔗 Step 7: Generating association rules...")
start_time = time.time()

# Optimization: High min_confidence for fewer rules
min_confidence = 0.9
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
rules_sorted = rules.sort_values(by='lift', ascending=False)

print(f"✅ Found {len(rules_sorted)} association rules (min_confidence={min_confidence})")
print_time("Association rules", start_time)

# === VISUALIZATION ===
print("📊 Step 8: Plotting top rules...")
start_time = time.time()

# Plot top 20 rules to reduce plotting time
top_rules = rules_sorted.head(20)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=top_rules, x='support', y='confidence', size='lift', hue='lift',
                palette='viridis', sizes=(100, 1000))
plt.title("Top 20 Association Rules from GSE2361 (FP-Growth)")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.grid(True)
plt.tight_layout()
plt.show()

print_time("Visualization", start_time)

# === SAVE OUTPUTS ===
print("💾 Step 9: Saving results to CSV files...")
start_time = time.time()

frequent_itemsets.to_csv("frequent_itemsets_gse2361.csv", index=False, compression=None)
rules_sorted.to_csv("association_rules_gse2361.csv", index=False, compression=None)

print("✅ All done! Results saved as:")
print("   → frequent_itemsets_gse2361.csv")
print("   → association_rules_gse2361.csv")
print_time("Saving outputs", start_time)
