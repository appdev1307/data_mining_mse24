import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json

# Force TkAgg backend for Matplotlib
plt.switch_backend('TkAgg')

# Function to log progress
def log_progress(text_widget, message):
    text_widget.config(state='normal')
    text_widget.insert(tk.END, f"{message}\n")
    text_widget.see(tk.END)
    text_widget.config(state='disabled')
    text_widget.update()

# Main analysis function
def run_analysis(min_support_entry, min_confidence_entry, progress_text, itemsets_tree, rules_tree, itemsets_plot_frame, rules_plot_frame):
    try:
        min_support = float(min_support_entry.get())
        min_confidence = float(min_confidence_entry.get())
        if not (0 < min_support <= 1) or not (0 < min_confidence <= 1):
            raise ValueError("min_support and min_confidence must be between 0 and 1.")
    except ValueError as e:
        messagebox.showerror("Input Error", f"Error: {e}. Please enter valid numbers.")
        return

    # Clear previous progress log
    progress_text.config(state='normal')
    progress_text.delete(1.0, tk.END)
    progress_text.config(state='disabled')

    log_progress(progress_text, "Loading and preprocessing data...")
    try:
        data = pd.read_csv('ai_job_dataset.csv')
    except FileNotFoundError:
        messagebox.showerror("File Error", "ai_job_dataset.csv not found in the script directory.")
        return
    transactions = data['required_skills'].str.lower().str.split(',').apply(lambda x: [item.strip() for item in x]).tolist()

    log_progress(progress_text, "Encoding transactions...")
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    log_progress(progress_text, f"Mining frequent itemsets with min_support={min_support}...")
    frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)

    log_progress(progress_text, "Filtering maximal frequent itemsets...")
    def is_maximal(itemset, frequent_itemsets):
        itemset_set = set(itemset)
        for other_itemset in frequent_itemsets['itemsets']:
            if set(other_itemset).issuperset(itemset_set) and other_itemset != itemset:
                return False
        return True

    maximal_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(
        lambda x: is_maximal(x, frequent_itemsets)
    )]

    log_progress(progress_text, f"Generating association rules with min_confidence={min_confidence}...")
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    log_progress(progress_text, "Saving results to CSV and text files...")
    maximal_itemsets_df = maximal_itemsets.copy()
    maximal_itemsets_df['itemsets'] = maximal_itemsets_df['itemsets'].apply(lambda x: ', '.join(list(x)))
    maximal_itemsets_df.to_csv('maximal_itemsets.csv', index=False)

    if not rules.empty:
        rules_df = rules.copy()
        rules_df['antecedents'] = rules_df['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules_df['consequents'] = rules_df['consequents'].apply(lambda x: ', '.join(list(x)))
        rules_df.to_csv('association_rules.csv', index=False)
    else:
        pd.DataFrame(columns=['antecedents', 'consequents', 'support', 'confidence', 'lift']).to_csv('association_rules.csv', index=False)

    with open('frequent_patterns_and_rules.txt', 'w') as f:
        f.write(f"Maximal Frequent Itemsets (min_support={min_support}):\n")
        for _, row in maximal_itemsets.iterrows():
            f.write(f"Itemset: {list(row['itemsets'])}, Support: {row['support']:.4f}\n")
        f.write(f"\nAssociation Rules (min_confidence={min_confidence}):\n")
        f.write(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_string())

    log_progress(progress_text, "Populating itemsets table...")
    for item in itemsets_tree.get_children():
        itemsets_tree.delete(item)
    for _, row in maximal_itemsets.iterrows():
        itemsets_tree.insert('', 'end', values=(', '.join(list(row['itemsets'])), f"{row['support']:.4f}"))

    log_progress(progress_text, "Populating rules table...")
    for item in rules_tree.get_children():
        rules_tree.delete(item)
    if rules.empty:
        rules_tree.insert('', 'end', values=("No rules generated.", "", "", "", ""))
    else:
        for _, row in rules.iterrows():
            rules_tree.insert('', 'end', values=(
                ', '.join(list(row['antecedents'])),
                ', '.join(list(row['consequents'])),
                f"{row['support']:.4f}",
                f"{row['confidence']:.4f}",
                f"{row['lift']:.4f}"
            ))

    log_progress(progress_text, "Rendering plots...")
    # Clear previous plots
    for widget in itemsets_plot_frame.winfo_children():
        widget.destroy()
    for widget in rules_plot_frame.winfo_children():
        widget.destroy()

    # Itemsets plot
    top_20_itemsets = maximal_itemsets.sort_values(by='support', ascending=False).head(20)
    itemset_labels = [', '.join(list(itemset)) for itemset in top_20_itemsets['itemsets']]
    itemset_fig_width = max(5, min(len(itemset_labels) * 0.3, 8))  # Dynamic width, capped at 8
    fig1, ax1 = plt.subplots(figsize=(itemset_fig_width, 2.5))
    ax1.bar(itemset_labels, top_20_itemsets['support'], color='skyblue', edgecolor='black')
    ax1.set_xlabel('Itemsets')
    ax1.set_ylabel('Support')
    ax1.set_title(f'Top 20 Itemsets (min_support={min_support})')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.tight_layout()
    canvas1 = FigureCanvasTkAgg(fig1, master=itemsets_plot_frame)
    canvas1.draw()
    canvas1.get_tk_widget().pack(fill='both', expand=True)
    canvas1.flush_events()

    # Rules plot
    rule_fig_width = max(5, min(len(rules) * 0.3, 8)) if not rules.empty else 5
    fig2, ax2 = plt.subplots(figsize=(rule_fig_width, 2.5))
    if not rules.empty:
        rule_labels = [f"{', '.join(list(rule['antecedents']))} → {', '.join(list(rule['consequents']))}" for _, rule in rules.iterrows()]
        ax2.bar(rule_labels, rules['confidence'], color='lightgreen', edgecolor='black')
        ax2.set_xlabel('Association Rules')
        ax2.set_ylabel('Confidence')
        ax2.set_title(f'Rules Confidence (min_confidence={min_confidence})')
        plt.xticks(rotation=45, ha='right', fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'No rules to plot', horizontalalignment='center', verticalalignment='center')
        ax2.set_xlabel('Association Rules')
        ax2.set_ylabel('Confidence')
        ax2.set_title(f'Rules Confidence (min_confidence={min_confidence})')
    plt.tight_layout()
    canvas2 = FigureCanvasTkAgg(fig2, master=rules_plot_frame)
    canvas2.draw()
    canvas2.get_tk_widget().pack(fill='both', expand=True)
    canvas2.flush_events()

    log_progress(progress_text, "Analysis complete.")

    # Chart.js configurations (for reference)
    chartjs_itemsets = {
        "type": "bar",
        "data": {
            "labels": itemset_labels,
            "datasets": [{
                "label": "Support",
                "data": top_20_itemsets['support'].tolist(),
                "backgroundColor": "#1f77b4",
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
                "title": { "display": True, "text": f"Top 20 Maximal Frequent Itemsets by Support (min_support={min_support})" }
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
                "backgroundColor": "#2ca02c",
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
                "title": { "display": True, "text": f"Confidence of Association Rules (min_confidence={min_confidence})" }
            }
        }
    }

    print(f"\nChart.js Configuration for Top 20 Maximal Frequent Itemsets (min_support={min_support}):")
    print(json.dumps(chartjs_itemsets, indent=2))
    print(f"\nChart.js Configuration for Association Rules (min_confidence={min_confidence}):")
    print(json.dumps(chartjs_rules, indent=2))

# GUI setup
root = tk.Tk()
root.title("Frequent Pattern Mining GUI")
root.geometry("1200x800")

# Input frame
input_frame = tk.Frame(root)
input_frame.pack(pady=5, padx=5, fill='x')

tk.Label(input_frame, text="Minimum Support (e.g., 0.02 for 2%):").grid(row=0, column=0, padx=5, pady=5)
min_support_entry = tk.Entry(input_frame)
min_support_entry.insert(0, "0.02")
min_support_entry.grid(row=0, column=1, padx=5, pady=5)

tk.Label(input_frame, text="Minimum Confidence (e.g., 0.2 for 20%):").grid(row=1, column=0, padx=5, pady=5)
min_confidence_entry = tk.Entry(input_frame)
min_confidence_entry.insert(0, "0.2")
min_confidence_entry.grid(row=1, column=1, padx=5, pady=5)

# Progress log
progress_frame = tk.LabelFrame(root, text="Progress Log")
progress_frame.pack(pady=5, padx=5, fill='both', expand=False)
progress_text = tk.Text(progress_frame, height=4, state='disabled')
progress_text.pack(fill='both', expand=True)

# Notebook for tabs
notebook = ttk.Notebook(root)
notebook.pack(pady=5, padx=5, fill='both', expand=True)

# Tables tab
tables_tab = tk.Frame(notebook)
notebook.add(tables_tab, text="Tables")

# Itemsets table
itemsets_frame = tk.LabelFrame(tables_tab, text="Maximal Frequent Itemsets")
itemsets_frame.pack(pady=5, padx=5, fill='both', expand=True)
itemsets_scrollbar_y = ttk.Scrollbar(itemsets_frame, orient='vertical')
itemsets_scrollbar_y.pack(side='right', fill='y')
itemsets_scrollbar_x = ttk.Scrollbar(itemsets_frame, orient='horizontal')
itemsets_scrollbar_x.pack(side='bottom', fill='x')
itemsets_tree = ttk.Treeview(itemsets_frame, columns=('Itemset', 'Support'), show='headings', yscrollcommand=itemsets_scrollbar_y.set, xscrollcommand=itemsets_scrollbar_x.set)
itemsets_tree.heading('Itemset', text='Itemset')
itemsets_tree.heading('Support', text='Support')
itemsets_tree.column('Itemset', width=400)
itemsets_tree.column('Support', width=100)
itemsets_tree.pack(fill='both', expand=True)
itemsets_scrollbar_y.config(command=itemsets_tree.yview)
itemsets_scrollbar_x.config(command=itemsets_tree.xview)

# Rules table
rules_frame = tk.LabelFrame(tables_tab, text="Association Rules")
rules_frame.pack(pady=5, padx=5, fill='both', expand=True)
rules_scrollbar_y = ttk.Scrollbar(rules_frame, orient='vertical')
rules_scrollbar_y.pack(side='right', fill='y')
rules_scrollbar_x = ttk.Scrollbar(rules_frame, orient='horizontal')
rules_scrollbar_x.pack(side='bottom', fill='x')
rules_tree = ttk.Treeview(rules_frame, columns=('Antecedents', 'Consequents', 'Support', 'Confidence', 'Lift'), show='headings', yscrollcommand=rules_scrollbar_y.set, xscrollcommand=rules_scrollbar_x.set)
rules_tree.heading('Antecedents', text='Antecedents')
rules_tree.heading('Consequents', text='Consequents')
rules_tree.heading('Support', text='Support')
rules_tree.heading('Confidence', text='Confidence')
rules_tree.heading('Lift', text='Lift')
rules_tree.column('Antecedents', width=200)
rules_tree.column('Consequents', width=200)
rules_tree.column('Support', width=100)
rules_tree.column('Confidence', width=100)
rules_tree.column('Lift', width=100)
rules_tree.pack(fill='both', expand=True)
rules_scrollbar_y.config(command=rules_tree.yview)
rules_scrollbar_x.config(command=rules_tree.xview)

# Plots tab
plots_tab = tk.Frame(notebook)
notebook.add(plots_tab, text="Plots")

# Itemsets plot frame
itemsets_plot_frame = tk.LabelFrame(plots_tab, text="Top 20 Maximal Frequent Itemsets Plot")
itemsets_plot_frame.pack(pady=5, padx=5, fill='both', expand=True)

# Rules plot frame
rules_plot_frame = tk.LabelFrame(plots_tab, text="Association Rules Confidence Plot")
rules_plot_frame.pack(pady=5, padx=5, fill='both', expand=True)

# Run button
run_button = tk.Button(input_frame, text="Run Analysis",
                       command=lambda: run_analysis(min_support_entry, min_confidence_entry,
                                                   progress_text, itemsets_tree, rules_tree,
                                                   itemsets_plot_frame, rules_plot_frame))
run_button.grid(row=2, column=0, columnspan=2, pady=5)

root.mainloop()
