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
def run_analysis(min_support_entry, min_confidence_entry, max_len_entry, progress_text, itemsets_tree, rules_tree, itemsets_plot_frame, rules_bar_plot_frame, rules_scatter_plot_frame, summary_text):
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

    log_progress(progress_text, f"Mining frequent itemsets with min_support={min_support}, max_len={max_len}...")
    frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True, max_len=max_len)

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
        f.write(f"Maximal Frequent Itemsets (min_support={min_support}, max_len={max_len}):\n")
        for _, row in maximal_itemsets.iterrows():
            f.write(f"Itemset: {list(row['itemsets'])}, Support: {row['support']:.4f}\n")
        f.write(f"\nAssociation Rules (min_confidence={min_confidence}):\n")
        f.write(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_string())

    log_progress(progress_text, "Populating itemsets table...")
    for _, row in maximal_itemsets.iterrows():
        itemsets_tree.insert('', 'end', values=(', '.join(list(row['itemsets'])), f"{row['support']:.4f}"))

    log_progress(progress_text, "Populating rules table...")
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

    log_progress(progress_text, "Populating summary...")
    summary_text.config(state='normal')
    summary_text.delete(1.0, tk.END)
    summary_text.insert(tk.END, "Frequent Pattern Mining Summary for AI Job Skills\n")
    summary_text.insert(tk.END, "=" * 50 + "\n\n")
    summary_text.insert(tk.END, "Dataset Context:\n")
    summary_text.insert(tk.END, "  Analyzed required skills from AI job postings to identify frequently occurring skill sets and their relationships.\n")
    summary_text.insert(tk.END, f"  Analysis parameters: min_support = {min_support:.2f}, min_confidence = {min_confidence:.2f}, max_itemset_size = {max_len}\n\n")
    summary_text.insert(tk.END, "Maximal Frequent Itemsets:\n")
    summary_text.insert(tk.END, f"  Total found: {len(maximal_itemsets)}\n")
    summary_text.insert(tk.END, f"  These itemsets represent skill combinations (up to {max_len} skills) that appear in at least "
                        f"{min_support*100:.1f}% of job postings and are not subsets of other frequent itemsets.\n")
    if not maximal_itemsets.empty:
        summary_text.insert(tk.END, "  Top 5 by support (indicating most common skill combinations):\n")
        top_itemsets = maximal_itemsets.sort_values(by='support', ascending=False).head(5)
        for i, (_, row) in enumerate(top_itemsets.iterrows(), 1):
            summary_text.insert(tk.END, f"    {i}. {', '.join(list(row['itemsets']))}: {row['support']*100:.1f}% of jobs require this skill set.\n")
        support_range = (maximal_itemsets['support'].min(), maximal_itemsets['support'].max())
        summary_text.insert(tk.END, f"  Support range: {support_range[0]*100:.1f}% to {support_range[1]*100:.1f}% of jobs.\n")
        max_size = maximal_itemsets['itemsets'].apply(len).max()
        summary_text.insert(tk.END, f"  Largest itemset size: {max_size} skills.\n")
    else:
        summary_text.insert(tk.END, "  No itemsets found meeting the support threshold.\n")
    summary_text.insert(tk.END, "\nAssociation Rules:\n")
    summary_text.insert(tk.END, f"  Total found: {len(rules)}\n")
    summary_text.insert(tk.END, "  These rules indicate strong skill co-occurrence patterns (if A, then B with high confidence).\n")
    if not rules.empty:
        summary_text.insert(tk.END, "  Top 5 by confidence (strongest skill associations):\n")
        top_rules = rules.sort_values(by='confidence', ascending=False).head(5)
        for i, (_, row) in enumerate(top_rules.iterrows(), 1):
            summary_text.insert(tk.END, f"    {i}. {', '.join(list(row['antecedents']))} → {', '.join(list(row['consequents']))}: "
                                       f"{row['confidence']*100:.1f}% confidence, {row['lift']:.2f}x more likely than random, "
                                       f"support {row['support']*100:.1f}%.\n")
        confidence_range = (rules['confidence'].min(), rules['confidence'].max())
        lift_range = (rules['lift'].min(), rules['lift'].max())
        summary_text.insert(tk.END, f"  Confidence range: {confidence_range[0]*100:.1f}% to {confidence_range[1]*100:.1f}%.\n")
        summary_text.insert(tk.END, f"  Lift range: {lift_range[0]:.2f}x to {lift_range[1]:.2f}x (values >1 indicate positive correlation).\n")
        max_lift_rule = rules.loc[rules['lift'].idxmax()] if len(rules) > 0 else None
        if max_lift_rule is not None:
            summary_text.insert(tk.END, f"  Strongest correlation: {', '.join(list(max_lift_rule['antecedents']))} → "
                                       f"{', '.join(list(max_lift_rule['consequents']))} (lift: {max_lift_rule['lift']:.2f}x).\n")
    else:
        summary_text.insert(tk.END, "  No rules generated meeting the confidence threshold.\n")
    summary_text.insert(tk.END, "\nKey Insights:\n")
    if not maximal_itemsets.empty:
        summary_text.insert(tk.END, f"  - Common skill combinations (up to {max_len} skills) reflect high-demand skills for AI jobs.\n")
    if not rules.empty:
        summary_text.insert(tk.END, "  - Strong rules suggest skill pairings critical for job roles, useful for prioritizing learning.\n")
    else:
        summary_text.insert(tk.END, "  - Try lowering min_support (e.g., 0.005) or increasing max_itemset_size to discover more patterns.\n")
    summary_text.config(state='disabled')

    log_progress(progress_text, "Rendering plots...")
    # Itemsets plot (in Plots tab)
    top_20_itemsets = maximal_itemsets.sort_values(by='support', ascending=False).head(20)
    itemset_labels = [', '.join(list(itemset)) for itemset in top_20_itemsets['itemsets']]
    itemset_fig_width = max(6, min(len(itemset_labels) * 0.4, 12))
    fig1, ax1 = plt.subplots(figsize=(itemset_fig_width, 4))
    ax1.bar(itemset_labels, top_20_itemsets['support'], color='skyblue', edgecolor='black')
    ax1.set_xlabel('Itemsets', fontsize=10)
    ax1.set_ylabel('Support', fontsize=10)
    ax1.set_title(f'Top 20 Itemsets (min_support={min_support})', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    canvas1 = FigureCanvasTkAgg(fig1, master=itemsets_plot_frame)
    canvas1.draw()
    canvas1.get_tk_widget().pack(fill='both', expand=True)
    canvas1.flush_events()

    # Rules plots (in Rules Plots tab)
    top_20_rules = rules.sort_values(by='confidence', ascending=False).head(20)
    
    # Bar plot: Top 20 rules by confidence
    rule_fig_width = max(6, min(len(top_20_rules) * 0.4, 12)) if not top_20_rules.empty else 6
    fig2, ax2 = plt.subplots(figsize=(rule_fig_width, 4))
    if not top_20_rules.empty:
        rule_labels = [f"Rule {i+1}" for i in range(len(top_20_rules))]  # Use Rule 1, Rule 2, ... for brevity
        ax2.bar(rule_labels, top_20_rules['confidence'], color='lightgreen', edgecolor='black')
        ax2.set_xlabel('Association Rules (See Tables tab for details)', fontsize=10)
        ax2.set_ylabel('Confidence', fontsize=10)
        ax2.set_title(f'Top 20 Rules by Confidence (min_confidence={min_confidence})', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'No rules to plot', horizontalalignment='center', verticalalignment='center', fontsize=10)
        ax2.set_xlabel('Association Rules', fontsize=10)
        ax2.set_ylabel('Confidence', fontsize=10)
        ax2.set_title(f'Top 20 Rules by Confidence (min_confidence={min_confidence})', fontsize=12)
    plt.tight_layout()
    canvas2 = FigureCanvasTkAgg(fig2, master=rules_bar_plot_frame)
    canvas2.draw()
    canvas2.get_tk_widget().pack(fill='both', expand=True)
    canvas2.flush_events()

    # Scatter plot: Support vs Lift, sized by Confidence
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    if not top_20_rules.empty:
        ax3.scatter(top_20_rules['support'], top_20_rules['lift'], 
                    s=top_20_rules['confidence'] * 500, alpha=0.6, c='purple')
        ax3.set_xlabel('Support', fontsize=10)
        ax3.set_ylabel('Lift', fontsize=10)
        ax3.set_title(f'Top 20 Rules: Support vs Lift (Size = Confidence)', fontsize=12)
        for i, (index, row) in enumerate(top_20_rules.iterrows()):
            ax3.annotate(f"{i+1}", (row['support'], row['lift']), fontsize=7, xytext=(5, 5), textcoords='offset points')
    else:
        ax3.text(0.5, 0.5, 'No rules to plot', horizontalalignment='center', verticalalignment='center', fontsize=10)
        ax3.set_xlabel('Support', fontsize=10)
        ax3.set_ylabel('Lift', fontsize=10)
        ax3.set_title(f'Top 20 Rules: Support vs Lift (Size = Confidence)', fontsize=12)
    plt.tight_layout()
    canvas3 = FigureCanvasTkAgg(fig3, master=rules_scatter_plot_frame)
    canvas3.draw()
    canvas3.get_tk_widget().pack(fill='both', expand=True)
    canvas3.flush_events()

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
                    "title": { "display": True, "text": "Support", "font": { "size": 12 } }
                },
                "x": {
                    "title": { "display": True, "text": "Itemsets", "font": { "size": 12 } }
                }
            },
            "plugins": {
                "legend": { "display": True, "labels": { "font": { "size": 12 } } },
                "title": { "display": True, "text": f"Top 20 Maximal Frequent Itemsets by Support (min_support={min_support})", "font": { "size": 14 } }
            }
        }
    }

    chartjs_rules_bar = {
        "type": "bar",
        "data": {
            "labels": rule_labels if not top_20_rules.empty else [],
            "datasets": [{
                "label": "Confidence",
                "data": top_20_rules['confidence'].tolist() if not top_20_rules.empty else [],
                "backgroundColor": "#2ca02c",
                "borderColor": "#ffffff",
                "borderWidth": 1
            }]
        },
        "options": {
            "scales": {
                "y": {
                    "beginAtZero": True,
                    "title": { "display": True, "text": "Confidence", "font": { "size": 12 } }
                },
                "x": {
                    "title": { "display": True, "text": "Association Rules (See Tables tab for details)", "font": { "size": 12 } }
                }
            },
            "plugins": {
                "legend": { "display": True, "labels": { "font": { "size": 12 } } },
                "title": { "display": True, "text": f"Top 20 Association Rules by Confidence (min_confidence={min_confidence})", "font": { "size": 14 } }
            }
        }
    }

    chartjs_rules_scatter = {
        "type": "scatter",
        "data": {
            "datasets": [{
                "label": "Rules",
                "data": [{"x": row['support'], "y": row['lift'], "r": row['confidence'] * 25} for _, row in top_20_rules.iterrows()] if not top_20_rules.empty else [],
                "backgroundColor": "purple",
                "borderColor": "#ffffff",
                "borderWidth": 1
            }]
        },
        "options": {
            "scales": {
                "y": {
                    "beginAtZero": True,
                    "title": { "display": True, "text": "Lift", "font": { "size": 12 } }
                },
                "x": {
                    "title": { "display": True, "text": "Support", "font": { "size": 12 } }
                }
            },
            "plugins": {
                "legend": { "display": True, "labels": { "font": { "size": 12 } } },
                "title": { "display": True, "text": f"Top 20 Rules: Support vs Lift (Size = Confidence)", "font": { "size": 14 } }
            }
        }
    }

    print(f"\nChart.js Configuration for Top 20 Maximal Frequent Itemsets (min_support={min_support}):")
    print(json.dumps(chartjs_itemsets, indent=2))
    print(f"\nChart.js Configuration for Top 20 Association Rules (Bar, min_confidence={min_confidence}):")
    print(json.dumps(chartjs_rules_bar, indent=2))
    print(f"\nChart.js Configuration for Top 20 Association Rules (Scatter, min_confidence={min_confidence}):")
    print(json.dumps(chartjs_rules_scatter, indent=2))

# GUI setup
root = tk.Tk()
root.title("Frequent Pattern Mining GUI")
root.geometry("1400x1000")

# Input frame
input_frame = tk.Frame(root)
input_frame.pack(pady=5, padx=5, fill='x')

tk.Label(input_frame, text="Minimum Support (e.g., 0.01 for 1%):", font=('Arial', 12)).grid(row=0, column=0, padx=5, pady=5)
min_support_entry = tk.Entry(input_frame, font=('Arial', 12))
min_support_entry.insert(0, "0.01")
min_support_entry.grid(row=0, column=1, padx=5, pady=5)

tk.Label(input_frame, text="Minimum Confidence (e.g., 0.1 for 10%):", font=('Arial', 12)).grid(row=1, column=0, padx=5, pady=5)
min_confidence_entry = tk.Entry(input_frame, font=('Arial', 12))
min_confidence_entry.insert(0, "0.1")
min_confidence_entry.grid(row=1, column=1, padx=5, pady=5)

tk.Label(input_frame, text="Max Itemset Size (e.g., 4):", font=('Arial', 12)).grid(row=2, column=0, padx=5, pady=5)
max_len_entry = tk.Entry(input_frame, font=('Arial', 12))
max_len_entry.insert(0, "4")
max_len_entry.grid(row=2, column=1, padx=5, pady=5)

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
style = ttk.Style()
style.configure("Treeview.Heading", font=('Arial', 12))
style.configure("Treeview", font=('Arial', 11))
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
notebook.add(plots_tab, text="Plots")

# Itemsets plot frame
itemsets_plot_frame = tk.LabelFrame(plots_tab, text="Top 20 Maximal Frequent Itemsets Plot", font=('Arial', 12))
itemsets_plot_frame.pack(pady=5, padx=5, fill='both', expand=True)

# Rules Plots tab
rules_plots_tab = tk.Frame(notebook)
notebook.add(rules_plots_tab, text="Rules Plots")

# Rules bar plot frame
rules_bar_plot_frame = tk.LabelFrame(rules_plots_tab, text="Top 20 Rules by Confidence", font=('Arial', 12))
rules_bar_plot_frame.pack(pady=5, padx=5, fill='both', expand=True)

# Rules scatter plot frame
rules_scatter_plot_frame = tk.LabelFrame(rules_plots_tab, text="Top 20 Rules: Support vs Lift", font=('Arial', 12))
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
                                                   itemsets_plot_frame, rules_bar_plot_frame, rules_scatter_plot_frame, summary_text))
run_button.grid(row=3, column=0, columnspan=2, pady=5)

root.mainloop()
