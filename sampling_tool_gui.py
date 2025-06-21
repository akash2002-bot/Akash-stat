
import pandas as pd
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, Toplevel, Label, Button
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys

# Initialize main window
root = tk.Tk()
root.title("Sampling Tool")
root.geometry("300x120")

# Help text
HELP_TEXT = """
📌 Sampling Methods Explanation:

1. Simple Random Sampling  
   Simple random sampling is a method of selecting a sample from a larger population where each member of the population has an equal and independent chance of being chosen.

2. Systematic Sampling  
   Systematic sampling is a probability sampling method where sample members are selected from a larger population according to a random starting point, but with a fixed, periodic interval.

3. Stratified Sampling  
   Stratified sampling is a statistical method where a population is divided into subgroups (strata) based on shared characteristics, and then a random sample is taken from each stratum.

4. Cluster Sampling  
   Cluster sampling is a probability sampling technique where the population is divided into groups, or "clusters," and then a random sample of these clusters is selected.

5. PPS Sampling  
   Probability Proportional to Size (PPS) sampling is a sampling technique where the probability of selecting a unit from a population is directly proportional to its size or some other measure considered relevant.
"""

def show_help():
    help_win = Toplevel(root)
    help_win.title("Help")
    help_win.geometry("600x500")
    Label(help_win, text=HELP_TEXT, justify='left', wraplength=580).pack(padx=10, pady=10)
    try:
        fig, ax = plt.subplots(figsize=(4,2))
        ax.bar(['Simple','Strat','Cluster','System','PPS'], [10]*5, color='skyblue')
        ax.set_title("Equal Sample Size Example")
        FigureCanvasTkAgg(fig, master=help_win).get_tk_widget().pack(pady=10)
    except:
        pass

def interrupted():
    messagebox.showwarning("Interrupted", "The process was interrupted.")
    sys.exit()

def load_file():
    fp = filedialog.askopenfilename(title="Select CSV file")
    if not fp: interrupted()
    try:
        return pd.read_csv(fp)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to read file:\n{e}")
        interrupted()

def choose_method():
    methods = ["Simple Random", "Stratified", "Cluster", "Systematic", "PPS", "All"]
    m = simpledialog.askstring("Sampling Method",
        f"Choose one: {', '.join(methods)}\nor type 'help'")
    if not m: interrupted()
    m = m.strip()
    if m.lower() == "help":
        show_help()
        return choose_method()
    if m not in methods:
        messagebox.showerror("Error", f"Unknown method: {m}")
        interrupted()
    return m

def get_size(N):
    s = simpledialog.askstring("Sample Size", f"Enter sample size (1-{N}):")
    if not s: interrupted()
    try:
        s = int(s)
        if not (1 <= s <= N): raise ValueError
        return s
    except:
        messagebox.showerror("Error", "Invalid sample size.")
        interrupted()

def simple_sample(df, n): return df.sample(n=n, random_state=42).reset_index(drop=True)

def stratified_sample(df, n):
    col = simpledialog.askstring("Strata Column", "Enter column for stratification:")
    if not col: interrupted()
    if col not in df.columns:
        messagebox.showerror("Error", f"Column '{col}' not found.")
        interrupted()
    grp = df.groupby(col)
    return grp.apply(lambda x: x.sample(n=max(1,int(round(n*len(x)/len(df)))), random_state=42)).reset_index(drop=True)

def cluster_sample(df, n):
    col = simpledialog.askstring("Cluster Column", "Enter column for clusters:")
    if not col: interrupted()
    if col not in df.columns:
        messagebox.showerror("Error", f"Column '{col}' not found.")
        interrupted()
    clusters = df[col].dropna().unique()
    sel = random.sample(list(clusters), min(len(clusters), n))
    return df[df[col].isin(sel)].reset_index(drop=True)

def systematic_sample(df, n):
    step = len(df)//n
    if step < 1: interrupted()
    start = random.randint(0, step-1)
    idx = list(range(start, len(df), step))[:n]
    return df.iloc[idx].reset_index(drop=True)

def pps_sample(df, n):
    col = simpledialog.askstring("PPS Column", "Enter PPS weight column:")
    if not col: interrupted()
    if col not in df.columns:
        messagebox.showerror("Error", f"Column '{col}' not found.")
        interrupted()
    auto = messagebox.askyesno("Auto PPS?", "Auto-calculate PPS weights?")
    if auto:
        df2 = df[df[col]>0].copy()
        df2['prob'] = df2[col]/df2[col].sum()
    else:
        bins = simpledialog.askstring("Bins", "Enter number of bins:")
        if not bins: interrupted()
        try:
            b = int(bins)
        except:
            messagebox.showerror("Error", "Invalid bins.")
            interrupted()
        labels = simpledialog.askstring("Labels", f"Enter {b} labels comma-separated:")
        if not labels: interrupted()
        lbls = [l.strip() for l in labels.split(',')]
        if len(lbls)!=b: 
            messagebox.showerror("Error", "Label count mismatch"); interrupted()
        ws = simpledialog.askstring("Weights", f"Enter {b} weights comma-separated:")
        if not ws: interrupted()
        wlst = [float(x) for x in ws.split(',')]
        if len(wlst)!=b: 
            messagebox.showerror("Error", "Weight count mismatch"); interrupted()
        df2 = df.copy()
        df2[col+'_bin'] = pd.qcut(df2[col], q=b, labels=lbls)
        m = dict(zip(lbls, wlst))
        df2['prob'] = df2[col+'_bin'].map(m)
        df2['prob'] /= df2['prob'].sum()
    try:
        return df2.sample(n=n, weights='prob', random_state=42).reset_index(drop=True)
    except Exception as e:
        messagebox.showerror("Error", f"PPS sampling failed:\n{e}")
        interrupted()

def preview_save(df, name):
    messagebox.showinfo(f"{name} Sample", df.head().to_string())
    sp = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV","*.csv")], title=f"Save {name}")
    if sp:
        df.to_csv(sp, index=False)
        print(f"✅ {name} saved to: {sp}")
    else:
        print(f"⚠ {name} save cancelled.")

def run_all(df, n):
    all_excel = messagebox.askyesno("Combined?", "Save all samples into one Excel?")
    if all_excel is None: interrupted()
    results = {}
    methods = [
        ("Simple Random", simple_sample),
        ("Stratified", stratified_sample),
        ("Cluster", cluster_sample),
        ("Systematic", systematic_sample),
        ("PPS", pps_sample),
    ]
    for name, func in methods:
        messagebox.showinfo("Running", f"Running {name} sample...")
        res = func(df, n)
        if all_excel:
            results[name] = res
        else:
            preview_save(res, name)
    if all_excel and results:
        xls = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel","*.xlsx")], title="Save all")
        if xls:
            with pd.ExcelWriter(xls) as w:
                for name, dfres in results.items():
                    dfres.to_excel(w, sheet_name=name[:31], index=False)
            messagebox.showinfo("Saved", f"Combined Excel saved to: {xls}")
        else:
            messagebox.showwarning("Cancelled", "Combined save cancelled.")

def run():
    df = load_file()
    m = choose_method()
    n = get_size(len(df))
    if m == "Simple Random":
        preview_save(simple_sample(df, n), m)
    elif m == "Stratified":
        preview_save(stratified_sample(df, n), m)
    elif m == "Cluster":
        preview_save(cluster_sample(df, n), m)
    elif m == "Systematic":
        preview_save(systematic_sample(df, n), m)
    elif m == "PPS":
        preview_save(pps_sample(df, n), m)
    elif m == "All":
        run_all(df, n)
    else:
        interrupted()

Button(root, text="Start Sampling", command=run, width=25).pack(pady=10)
Button(root, text="Help", command=show_help, width=25).pack(pady=5)

root.mainloop()
