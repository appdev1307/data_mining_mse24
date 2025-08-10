#
#
# conda activate dam501_env
#
#


# MaxFP-Growth GUI
# Updated version of your Tkinter GUI that implements a MaxFP-Growth-like
# mining strategy: it builds an FP-tree and mines frequent itemsets while
# applying maximality pruning during the mining (skip itemsets that are
# subset of already discovered maximal itemsets and remove subsumed ones).
#
# This implementation keeps the original GUI layout and CSV outputs.
# It replaces the use of mlxtend.fpgrowth with a local FP-tree + mining.

import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
from collections import defaultdict, Counter

# Force TkAgg backend for Matplotlib
plt.switch_backend('TkAgg')


# ----------------------------- FP-tree classes -----------------------------
class FPTreeNode:
    def __init__(self, item_name, count, parent):
        self.item_name = item_name
        self.count = count
        self.parent = parent
        self.children = {}
        self.node_link = None  # link to next node with same item_name

    def increment(self, count):
        self.count += count


class FPTree:
    def __init__(self, transactions, min_support_count):
        self.root = FPTreeNode(None, 1, None)
        self.header_table = {}  # item -> [support, head_of_node_link]
        self.min_support_count = min_support_count
        self.build_header_table(transactions)
        self.build_tree(transactions)

    def build_header_table(self, transactions):
        # Count frequency of items
        item_counter = Counter()
        for tran in transactions:
            item_counter.update(tran)
        # Keep only frequent items
        for item, count in list(item_counter.items()):
            if count >= self.min_support_count:
                self.header_table[item] = [count, None]
        # Sort header table by descending frequency (helps create compact tree)
        self.header_order = [item for item, _ in sorted(self.header_table.items(), key=lambda x: (-x[1][0], x[0]))]

    def build_tree(self, transactions):
        for tran in transactions:
            # Filter and order transaction items by header_order
            ordered_items = [item for item in self.header_order if item in tran]
            if ordered_items:
                self._insert_tree(ordered_items, self.root)

    def _insert_tree(self, items, node):
        first = items[0]
        # if child exists, increment, else create
        if first in node.children:
            node.children[first].increment(1)
            child = node.children[first]
        else:
            child = FPTreeNode(first, 1, node)
            node.children[first] = child
            # update header table node link
            if self.header_table[first][1] is None:
                self.header_table[first][1] = child
            else:
                # attach at end of node-link chain
                current = self.header_table[first][1]
                while current.node_link is not None:
                    current = current.node_link
                current.node_link = child
        # recursively add remaining
        remaining = items[1:]
        if remaining:
            self._insert_tree(remaining, child)

    def conditional_pattern_base(self, item):
        # traverse node links for item and collect prefix paths
        base_patterns = []  # list of (prefix_path_list, count)
        node = self.header_table[item][1]
        while node is not None:
            count = node.count
            path = []
            parent = node.parent
            while parent is not None and parent.item_name is not None:
                path.append(parent.item_name)
                parent = parent.parent
            if path:
                base_patterns.append((list(reversed(path)), count))
            node = node.node_link
        return base_patterns

    def has_single_path(self):
        # Check if tree is a single path (no branching)
        node = self.root
        while True:
            if len(node.children) > 1:
                return False
            elif len(node.children) == 0:
                return True
            node = list(node.children.values())[0]


# --------------------------- MaxFP-Growth miner ----------------------------
class MaxFPGrowth:
    def __init__(self, transactions, min_support, min_support_count=None):
        self.transactions = transactions
        self.min_support = min_support
        self.min_support_count = min_support_count
        self.maximal_itemsets = {}  # frozenset -> support_count

    def run(self):
        # If min_support_count not provided, compute from transactions
        if self.min_support_count is None:
            self.min_support_count = max(1, int(self.min_support * len(self.transactions)))

        # build initial FP-tree
        tree = FPTree(self.transactions, self.min_support_count)
        # header ordered by increasing support (for mining in fp-growth classic we go lowest first)
        # but for maximal pruning efficiency we'll mine items in ascending frequency (standard strategy)
        ordered_items = [item for item, _ in sorted(tree.header_table.items(), key=lambda kv: (kv[1][0], kv[0]))]

        # recursively mine
        self._mine_tree(tree, suffix=frozenset())
        # convert counts to supports
        results = []
        n = len(self.transactions)
        for itemset, cnt in self.maximal_itemsets.items():
            results.append({'itemset': set(itemset), 'support': cnt / n})
        return pd.DataFrame(results)

    def _is_subsumed_by_existing_maximal(self, candidate):
        # if any existing maximal_itemset is a superset of candidate then candidate is not maximal
        for m in self.maximal_itemsets.keys():
            if set(candidate).issubset(set(m)):
                return True
        return False

    def _remove_subsumed_maximals(self, candidate, support_count):
        # If candidate is a superset of any existing maximals, remove those (they are no longer maximal)
        to_remove = []
        for m in list(self.maximal_itemsets.keys()):
            if set(m).issubset(set(candidate)):
                to_remove.append(m)
        for r in to_remove:
            del self.maximal_itemsets[r]
        # add candidate
        self.maximal_itemsets[frozenset(candidate)] = support_count

    def _mine_tree(self, tree, suffix):
        # If tree is empty or has no header items, stop
        if not tree.header_table:
            return

        # If tree has a single path, generate combinations and apply maximal pruning
        if tree.has_single_path():
            # collect path items with counts
            path_nodes = []
            node = tree.root
            while len(node.children) == 1:
                node = list(node.children.values())[0]
                path_nodes.append((node.item_name, node.count))
            # generate all combinations of path items (power set)
            items = [item for item, cnt in path_nodes]
            counts = {item: cnt for item, cnt in path_nodes}
            # generate combinations in descending length to favor maximal sets earlier
            from itertools import combinations
            for r in range(len(items), 0, -1):
                for comb in combinations(items, r):
                    cand = set(comb) | set(suffix)
                    # support of the combination is min count in the path for items in comb
                    support_count = min([counts[i] for i in comb]) if comb else 0
                    # primal maximality check
                    if not self._is_subsumed_by_existing_maximal(cand):
                        self._remove_subsumed_maximals(cand, support_count)
            return

        # Otherwise, for each item (process in ascending support order)
        # build conditional tree and recurse
        # Sort header items by increasing support count to follow FP-growth mining order
        items_sorted = [item for item, _ in sorted(tree.header_table.items(), key=lambda kv: (kv[1][0], kv[0]))]

        for item in items_sorted:
            support_count = tree.header_table[item][0]
            new_suffix = set(suffix)
            new_suffix.add(item)
            # primal maximal check: if suffix is already subsumed, skip
            if self._is_subsumed_by_existing_maximal(new_suffix):
                continue
            # add candidate provisionally (we'll remove subsumed later if it's actually maximal)
            # But before adding we should ensure its support count meets threshold (it does by construction)
            # obtain conditional pattern base and build conditional FP-tree
            base_patterns = tree.conditional_pattern_base(item)
            # build transaction list for conditional tree by replicating prefix with counts
            cond_transactions = []
            for prefix, cnt in base_patterns:
                # replicate prefix cnt times is heavy; instead we append prefix with multiplicity via count field
                # but our FPTree constructor expects list of lists; replicate for simplicity (prefix sizes are small)
                for _ in range(cnt):
                    cond_transactions.append(prefix)
            if not cond_transactions:
                # leaf node - candidate is maximal
                # add candidate and remove subsumed existing maximals
                self._remove_subsumed_maximals(new_suffix, support_count)
                continue
            cond_tree = FPTree(cond_transactions, self.min_support_count)
            # If cond_tree.header_table is empty -> no further items, treat new_suffix as maximal
            if not cond_tree.header_table:
                self._remove_subsumed_maximals(new_suffix, support_count)
            else:
                # recurse: mine conditional tree with updated suffix
                self._mine_tree(cond_tree, frozenset(new_suffix))
                # After mining conditional tree, it's possible that no maximal superset including 'item' was added
                # If so, the combination (item + suffix) might still be maximal; check and add
                if not self._is_subsumed_by_existing_maximal(new_suffix):
                    # support_count defined earlier
                    self._remove_subsumed_maximals(new_suffix, support_count)


# ---------------------------- Utility functions ----------------------------

def preprocess_transactions_from_csv(path):
    data = pd.read_csv(path)
    if 'required_skills' not in data.columns:
        raise ValueError("CSV must contain a 'required_skills' column")
    transactions = data['required_skills'].astype(str).str.lower().str.split(',').apply(lambda x: [i.strip() for i in x if i and i.strip()]).tolist()
    return transactions, data


# ----------------------------- GUI and plotting ---------------------------
# Function to log progress

def log_progress(text_widget, message):
    text_widget.config(state='normal')
    text_widget.insert(tk.END, f"{message}\n")
    text_widget.see(tk.END)
    text_widget.config(state='disabled')
    text_widget.update()


# Main analysis function (uses MaxFPGrowth)

def run_analysis(min_support_entry, min_confidence_entry, max_len_entry, progress_text, itemsets_tree, rules_tree, itemsets_plot_frame, rules_bar_plot_frame, rules_scatter_plot_frame, summary_text, dataset_path_var):
    try:
        min_support = float(min_support_entry.get())
        min_confidence = float(min_confidence_entry.get())
        max_len = int(max_len_entry.get())
        if not (0 < min_support <= 1) or not (0 < min_confidence <= 1):
            raise ValueError("min_support and min_confidence must be between 0 and 1.")
        if max_len < 1:
            raise ValueError("max_len must be at least 1.")
    except ValueError as e:
        messagebox.showerror("Input Error", f"Error: {e}. Please enter valid numbers.")
        return

    # Clear previous content
    progress_text.config(state='normal')
    progress_text.delete(1.0, tk.END)
    progress_text.config(state='disabled')
    summary_text.config(state='normal')
    summary_text.delete(1.0, tk.END)
    summary_text.config(state='disabled')
    for item in itemsets_tree.get_children():
        itemsets_tree.delete(item)
    for item in rules_tree.get_children():
        rules_tree.delete(item)
    for widget in itemsets_plot_frame.winfo_children():
        widget.destroy()
    for widget in rules_bar_plot_frame.winfo_children():
        widget.destroy()
    for widget in rules_scatter_plot_frame.winfo_children():
        widget.destroy()

    dataset_path = dataset_path_var.get()
    if not dataset_path:
        messagebox.showerror("File Error", "No dataset selected. Please choose ai_job_dataset.csv or another dataset.")
        return

    log_progress(progress_text, "Loading and preprocessing data...")
    try:
        transactions, raw_df = preprocess_transactions_from_csv(dataset_path)
    except Exception as e:
        messagebox.showerror("File Error", str(e))
        return

    n_transactions = len(transactions)
    log_progress(progress_text, f"Total transactions: {n_transactions}")

    log_progress(progress_text, "Running MaxFP-Growth mining...")
    miner = MaxFPGrowth(transactions, min_support)
    maximal_itemsets_df = miner.run()

    # Filter by max_len value
    maximal_itemsets_df['size'] = maximal_itemsets_df['itemset'].apply(len)
    maximal_itemsets_df = maximal_itemsets_df[maximal_itemsets_df['size'] <= max_len]
    maximal_itemsets_df = maximal_itemsets_df.sort_values(by='support', ascending=False)

    log_progress(progress_text, f"Found {len(maximal_itemsets_df)} maximal itemsets (after max_len filter)")

    # Save maximal itemsets
    maximal_itemsets_df_out = maximal_itemsets_df.copy()
    maximal_itemsets_df_out['itemset'] = maximal_itemsets_df_out['itemset'].apply(lambda s: ', '.join(sorted(list(s))))
    maximal_itemsets_df_out[['itemset', 'support']].to_csv('maximal_itemsets.csv', index=False)

    # For association rules we will compute rules from the maximal itemsets by checking subsets
    # NOTE: association rules from only maximal itemsets are incomplete vs using all frequent itemsets,
    # but for efficiency we will generate candidate rules from maximal itemsets' subsets.

    log_progress(progress_text, "Generating association rules from maximal itemsets (approximate)...")
    rules_rows = []
    for _, row in maximal_itemsets_df.iterrows():
        items = list(row['itemset'])
        support_itemset = row['support']
        # generate non-empty proper subsets as antecedents
        from itertools import chain, combinations
        def subsets(iterable):
            s = list(iterable)
            return chain.from_iterable(combinations(s, r) for r in range(1, len(s)))
        for antecedent in subsets(items):
            antecedent = set(antecedent)
            consequent = set(items) - antecedent
            if not consequent:
                continue
            # compute support of antecedent by scanning transactions (could be optimized with support cache)
            antecedent_count = 0
            union_count = 0
            for t in transactions:
                tset = set(t)
                if antecedent.issubset(tset):
                    antecedent_count += 1
                    if consequent.issubset(tset):
                        union_count += 1
            if antecedent_count == 0:
                continue
            confidence = union_count / antecedent_count
            support = union_count / n_transactions
            if confidence >= min_confidence:
                lift = (confidence) / ( (sum(1 for t in transactions if set(consequent).issubset(set(t))) / n_transactions) or 1e-9 )
                rules_rows.append({
                    'antecedents': ', '.join(sorted(antecedent)),
                    'consequents': ', '.join(sorted(consequent)),
                    'support': support,
                    'confidence': confidence,
                    'lift': lift
                })

    rules_df = pd.DataFrame(rules_rows)
    if rules_df.empty:
        pd.DataFrame(columns=['antecedents','consequents','support','confidence','lift']).to_csv('association_rules.csv', index=False)
    else:
        rules_df = rules_df.sort_values(by='confidence', ascending=False)
        rules_df.to_csv('association_rules.csv', index=False)

    # save textual report
    with open('frequent_patterns_and_rules.txt', 'w') as f:
        f.write(f"Maximal Frequent Itemsets (min_support={min_support}, max_len={max_len}):\n")
        for _, row in maximal_itemsets_df.iterrows():
            f.write(f"Itemset: {sorted(list(row['itemset']))}, Support: {row['support']:.4f}\n")
        f.write('\nAssociation Rules (approx, from maximal sets):\n')
        if not rules_df.empty:
            f.write(rules_df.to_string(index=False))

    # populate itemsets table
    log_progress(progress_text, "Populating itemsets table...")
    for _, row in maximal_itemsets_df.iterrows():
        itemsets_tree.insert('', 'end', values=(', '.join(sorted(list(row['itemset']))), f"{row['support']:.4f}"))

    # populate rules table
    log_progress(progress_text, "Populating rules table...")
    if rules_df.empty:
        rules_tree.insert('', 'end', values=("No rules generated.", "", "", "", ""))
    else:
        for _, row in rules_df.iterrows():
            rules_tree.insert('', 'end', values=(row['antecedents'], row['consequents'], f"{row['support']:.4f}", f"{row['confidence']:.4f}", f"{row['lift']:.4f}"))

    # summary
    log_progress(progress_text, "Populating summary...")
    summary_text.config(state='normal')
    summary_text.delete(1.0, tk.END)
    summary_text.insert(tk.END, "Frequent Pattern Mining Summary for AI Job Skills\n")
    summary_text.insert(tk.END, "="*50 + "\n\n")
    summary_text.insert(tk.END, "Dataset Context:\n")
    summary_text.insert(tk.END, "  Analyzed required skills from AI job postings to identify frequently occurring skill sets and their relationships.\n")
    summary_text.insert(tk.END, f"  Analysis parameters: min_support = {min_support:.2f}, min_confidence = {min_confidence:.2f}, max_itemset_size = {max_len}\n\n")
    summary_text.insert(tk.END, "Maximal Frequent Itemsets:\n")
    summary_text.insert(tk.END, f"  Total found: {len(maximal_itemsets_df)}\n")
    if not maximal_itemsets_df.empty:
        summary_text.insert(tk.END, "  Top 5 by support (indicating most common skill combinations):\n")
        top_itemsets = maximal_itemsets_df.head(5)
        for i, (_, row) in enumerate(top_itemsets.iterrows(), 1):
            summary_text.insert(tk.END, f"    {i}. {', '.join(sorted(list(row['itemset'])))}: {row['support']*100:.1f}% of jobs require this skill set.\n")
        support_range = (maximal_itemsets_df['support'].min(), maximal_itemsets_df['support'].max())
        summary_text.insert(tk.END, f"  Support range: {support_range[0]*100:.1f}% to {support_range[1]*100:.1f}% of jobs.\n")
        max_size = maximal_itemsets_df['itemset'].apply(len).max()
        summary_text.insert(tk.END, f"  Largest itemset size: {max_size} skills.\n")
    else:
        summary_text.insert(tk.END, "  No itemsets found meeting the support threshold.\n")

    summary_text.insert(tk.END, "\nAssociation Rules:\n")
    summary_text.insert(tk.END, f"  Total found: {len(rules_df)}\n")
    if not rules_df.empty:
        top_rules = rules_df.head(5)
        summary_text.insert(tk.END, "  Top 5 by confidence (strongest skill associations):\n")
        for i, (_, row) in enumerate(top_rules.iterrows(), 1):
            summary_text.insert(tk.END, f"    {i}. {row['antecedents']} → {row['consequents']}: {row['confidence']*100:.1f}% confidence, {row['lift']:.2f}x, support {row['support']*100:.1f}%.\n")
    summary_text.config(state='disabled')

    log_progress(progress_text, "Rendering plots...")

    # plotting (top itemsets)
    plot_width = 10
    plot_height = 4
    top_20_itemsets = maximal_itemsets_df.head(20)
    itemset_labels = [', '.join(sorted(list(x))) for x in top_20_itemsets['itemset']]

    fig1, ax1 = plt.subplots(figsize=(plot_width, plot_height))
    ax1.bar(itemset_labels, top_20_itemsets['support'])
    ax1.set_xlabel('Itemsets', fontsize=10)
    ax1.set_ylabel('Support', fontsize=10)
    ax1.set_title(f'Top 20 Itemsets (min_support={min_support:.2f})', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    canvas1 = FigureCanvasTkAgg(fig1, master=itemsets_plot_frame)
    canvas1.draw()
    canvas1.get_tk_widget().pack(fill='both', expand=True)

    # rules bar
    top_20_rules = rules_df.head(20) if not rules_df.empty else pd.DataFrame()
    fig2, ax2 = plt.subplots(figsize=(plot_width, plot_height))
    if not top_20_rules.empty:
        rule_labels = [f"Rule {i+1}" for i in range(len(top_20_rules))]
        ax2.bar(rule_labels, top_20_rules['confidence'])
        ax2.set_ylim(0, 1)
        plt.xticks(rotation=45, ha='right', fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'No rules to plot', horizontalalignment='center', verticalalignment='center')
    canvas2 = FigureCanvasTkAgg(fig2, master=rules_bar_plot_frame)
    canvas2.draw()
    canvas2.get_tk_widget().pack(fill='both', expand=True)

    # scatter support vs lift
    fig3, ax3 = plt.subplots(figsize=(plot_width, plot_height))
    if not top_20_rules.empty:
        ax3.scatter(top_20_rules['support'], top_20_rules['lift'], s=top_20_rules['confidence']*200, alpha=0.6)
        ax3.set_xlabel('Support')
        ax3.set_ylabel('Lift')
    else:
        ax3.text(0.5, 0.5, 'No rules to plot', horizontalalignment='center', verticalalignment='center')
    canvas3 = FigureCanvasTkAgg(fig3, master=rules_scatter_plot_frame)
    canvas3.draw()
    canvas3.get_tk_widget().pack(fill='both', expand=True)

    log_progress(progress_text, 'Analysis complete.')


# GUI setup (same layout as original, with small additions)
root = tk.Tk()
root.title("MaxFP-Growth Frequent Pattern Mining GUI")
root.geometry("1400x1000")

# Input frame
input_frame = tk.Frame(root)
input_frame.pack(pady=5, padx=5, fill='x')

# Dataset selection
dataset_path_var = tk.StringVar()

def browse_dataset():
    path = filedialog.askopenfilename(title='Select dataset CSV', filetypes=[('CSV files', '*.csv'), ('All files', '*.*')])
    if path:
        dataset_path_var.set(path)

browse_btn = tk.Button(input_frame, text='Choose CSV...', command=browse_dataset)
browse_btn.grid(row=0, column=0, padx=5, pady=5)

dataset_label = tk.Label(input_frame, textvariable=dataset_path_var)
dataset_label.grid(row=0, column=1, columnspan=4, sticky='w')

# parameters
tk.Label(input_frame, text="Minimum Support (e.g., 0.01 for 1%):", font=('Arial', 12)).grid(row=1, column=0, padx=5, pady=5)
min_support_entry = tk.Entry(input_frame, font=('Arial', 12))
min_support_entry.insert(0, "0.01")
min_support_entry.grid(row=1, column=1, padx=5, pady=5)

tk.Label(input_frame, text="Minimum Confidence (e.g., 0.1 for 10%):", font=('Arial', 12)).grid(row=2, column=0, padx=5, pady=5)
min_confidence_entry = tk.Entry(input_frame, font=('Arial', 12))
min_confidence_entry.insert(0, "0.1")
min_confidence_entry.grid(row=2, column=1, padx=5, pady=5)

tk.Label(input_frame, text="Max Itemset Size (e.g., 4):", font=('Arial', 12)).grid(row=3, column=0, padx=5, pady=5)
max_len_entry = tk.Entry(input_frame, font=('Arial', 12))
max_len_entry.insert(0, "4")
max_len_entry.grid(row=3, column=1, padx=5, pady=5)

# Progress log
progress_frame = tk.LabelFrame(root, text="Progress Log", font=('Arial', 12))
progress_frame.pack(pady=5, padx=5, fill='both', expand=False)
progress_text = tk.Text(progress_frame, height=4, state='disabled', font=('Arial', 10))
progress_text.pack(fill='both', expand=True)

# Notebook for tabs
notebook = ttk.Notebook(root)
notebook.pack(pady=5, padx=5, fill='both', expand=True)

# Tables tab
tables_tab = tk.Frame(notebook)
notebook.add(tables_tab, text="Tables")

# Itemsets table
itemsets_frame = tk.LabelFrame(tables_tab, text="Maximal Frequent Itemsets", font=('Arial', 12))
itemsets_frame.pack(pady=5, padx=5, fill='both', expand=True)
itemsets_scrollbar_y = ttk.Scrollbar(itemsets_frame, orient='vertical')
itemsets_scrollbar_y.pack(side='right', fill='y')
itemsets_scrollbar_x = ttk.Scrollbar(itemsets_frame, orient='horizontal')
itemsets_scrollbar_x.pack(side='bottom', fill='x')
itemsets_tree = ttk.Treeview(itemsets_frame, columns=('Itemset', 'Support'), show='headings', yscrollcommand=itemsets_scrollbar_y.set, xscrollcommand=itemsets_scrollbar_x.set)
itemsets_tree.heading('Itemset', text='Itemset', anchor='w')
itemsets_tree.heading('Support', text='Support', anchor='w')
itemsets_tree.column('Itemset', width=600)
itemsets_tree.column('Support', width=150)
itemsets_tree.pack(fill='both', expand=True)
itemsets_scrollbar_y.config(command=itemsets_tree.yview)
itemsets_scrollbar_x.config(command=itemsets_tree.xview)

# Rules table
rules_frame = tk.LabelFrame(tables_tab, text="Association Rules", font=('Arial', 12))
rules_frame.pack(pady=5, padx=5, fill='both', expand=True)
rules_scrollbar_y = ttk.Scrollbar(rules_frame, orient='vertical')
rules_scrollbar_y.pack(side='right', fill='y')
rules_scrollbar_x = ttk.Scrollbar(rules_frame, orient='horizontal')
rules_scrollbar_x.pack(side='bottom', fill='x')
rules_tree = ttk.Treeview(rules_frame, columns=('Antecedents', 'Consequents', 'Support', 'Confidence', 'Lift'), show='headings', yscrollcommand=rules_scrollbar_y.set, xscrollcommand=rules_scrollbar_x.set)
rules_tree.heading('Antecedents', text='Antecedents', anchor='w')
rules_tree.heading('Consequents', text='Consequents', anchor='w')
rules_tree.heading('Support', text='Support', anchor='w')
rules_tree.heading('Confidence', text='Confidence', anchor='w')
rules_tree.heading('Lift', text='Lift', anchor='w')
rules_tree.column('Antecedents', width=300)
rules_tree.column('Consequents', width=300)
rules_tree.column('Support', width=150)
rules_tree.column('Confidence', width=150)
rules_tree.column('Lift', width=150)
rules_tree.pack(fill='both', expand=True)
rules_scrollbar_y.config(command=rules_tree.yview)
rules_scrollbar_x.config(command=rules_tree.xview)

# Plots tab (only itemsets)
plots_tab = tk.Frame(notebook)
notebook.add(plots_tab, text="Itemsets Plot")

# Itemsets plot frame
itemsets_plot_frame = tk.LabelFrame(plots_tab, text="Top 20 Maximal Frequent Itemsets Plot", font=('Arial', 12))
itemsets_plot_frame.pack(pady=5, padx=5, fill='both', expand=True)

# Rules Bar Plot tab
rules_bar_tab = tk.Frame(notebook)
notebook.add(rules_bar_tab, text="Rules Bar Plot")

# Rules bar plot frame
rules_bar_plot_frame = tk.LabelFrame(rules_bar_tab, text="Top 20 Rules by Confidence", font=('Arial', 12))
rules_bar_plot_frame.pack(pady=5, padx=5, fill='both', expand=True)

# Rules Scatter Plot tab
rules_scatter_tab = tk.Frame(notebook)
notebook.add(rules_scatter_tab, text="Rules Scatter Plot")

# Rules scatter plot frame
rules_scatter_plot_frame = tk.LabelFrame(rules_scatter_tab, text="Top 20 Rules: Support vs Lift", font=('Arial', 12))
rules_scatter_plot_frame.pack(pady=5, padx=5, fill='both', expand=True)

# Summary tab
summary_tab = tk.Frame(notebook)
notebook.add(summary_tab, text="Summary")

# Summary text
summary_frame = tk.LabelFrame(summary_tab, text="Results Summary", font=('Arial', 12))
summary_frame.pack(pady=5, padx=5, fill='both', expand=True)
summary_scrollbar_y = ttk.Scrollbar(summary_frame, orient='vertical')
summary_scrollbar_y.pack(side='right', fill='y')
summary_text = tk.Text(summary_frame, height=10, wrap='word', yscrollcommand=summary_scrollbar_y.set, font=('Arial', 11))
summary_text.pack(fill='both', expand=True)
summary_scrollbar_y.config(command=summary_text.yview)
summary_text.config(state='disabled')

# Run button
run_button = tk.Button(input_frame, text="Run Analysis", font=('Arial', 12),
                       command=lambda: run_analysis(min_support_entry, min_confidence_entry, max_len_entry,
                                                   progress_text, itemsets_tree, rules_tree,
                                                   itemsets_plot_frame, rules_bar_plot_frame, rules_scatter_plot_frame, summary_text, dataset_path_var))
run_button.grid(row=4, column=0, columnspan=2, pady=5)

root.mainloop()
