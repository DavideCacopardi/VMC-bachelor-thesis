import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tkinter as tk
from tkinter import messagebox

def readmetadata(fname):
    meta_dict = {"Description": ""}
    if not os.path.exists(fname):
        return meta_dict
        
    on = False
    lines = ""
    with open(fname) as file:
        for line in file:
            idx = line.find("Hamiltonian")
            if idx != -1:
                on = True
            if on and line.find("--") == -1:
                lines += line
            idx = line.find("Called")
            if idx != -1:
                on = False
                break
    meta_dict["Description"] = lines
    return meta_dict

def plot_files(fnames, custom_save_name, talk=True):
    fdir = "./logs_OBD"
    
    fig1, axs1 = plt.subplots(1, 2, figsize=(12, 6))
    fig2, axs2 = plt.subplots(1, 2, figsize=(12, 6))
    
    cmap_colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(fnames))))
    has_pcorr = False
    
    for i, fname in enumerate(fnames):
        totfname = f"{fdir}/{fname}"
        log_fname = f"logs/{fname[:-4]}.log"
        md = readmetadata(log_fname)
        
        print(f"Loading and plotting: {totfname}")
        
        try:
            df = pd.read_csv(totfname)
            df.columns = [c.strip().replace('#', '').strip() for c in df.columns]
            df = df.dropna(axis=1, how='all')

            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.replace([-np.nan, np.nan, "nan", "-nan"], 0)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load {fname}:\n{e}")
            continue
            
        if 'r' not in df.columns:
            print(f"Skipping {fname}: column 'r' not found.")
            continue
            
        r = df['r']
        # e.g. '20260819_234333'
        label = fname[4:-4]
        base_color = cmap_colors[i % 10]
        
        # FIGURE 1: One-Body Density
        if 'dens_tot' in df.columns:
            err_col = 'dens_err_' if 'dens_err_' in df.columns else df.columns[2]
            axs1[0].errorbar(r, df['dens_tot'], yerr=df[err_col], 
                             color=base_color, marker='.', markersize=6, linestyle='-', label=f"{label} (tot)")
            
            if 'dens_A' in df.columns and len(fnames) == 1:
                axs1[0].plot(r, df['dens_A'], color=base_color, linestyle='--', alpha=0.5, label='A')
            if 'dens_B' in df.columns and len(fnames) == 1:
                axs1[0].plot(r, df['dens_B'], color=base_color, linestyle=':', alpha=0.5, label='B')
                
        if 'prob_tot' in df.columns:
            err_col_prob = 'prob_err_' if 'prob_err_' in df.columns else [c for c in df.columns if 'prob_err' in c][0]
            axs1[1].errorbar(r, df['prob_tot'], yerr=df[err_col_prob], 
                             color=base_color, marker='.', markersize=6, linestyle='-', label=f"{label} (tot)")
                             
            if 'prob_A' in df.columns and len(fnames) == 1:
                axs1[1].plot(r, df['prob_A'], color=base_color, linestyle='--', alpha=0.5, label='A')
            if 'prob_B' in df.columns and len(fnames) == 1:
                axs1[1].plot(r, df['prob_B'], color=base_color, linestyle=':', alpha=0.5, label='B')

        # FIGURE 2: Pair Correlation (alike / unlike)
        if 'dens_alike' in df.columns:
            has_pcorr = True
            axs2[0].errorbar(r, df['dens_alike'], yerr=df['dens_alike_err'], 
                             color=base_color, marker='.', markersize=6, linestyle='-', label=f"{label} (alike)")
            axs2[0].errorbar(r, df['dens_unlike'], yerr=df['dens_unlike_err'], 
                             color=base_color, marker='x', markersize=6, linestyle='--', label=f"{label} (unlike)")
            
            axs2[1].errorbar(r, df['prob_alike'], yerr=df['prob_alike_err'], 
                             color=base_color, marker='.', markersize=6, linestyle='-', label=f"{label} (alike)")
            axs2[1].errorbar(r, df['prob_unlike'], yerr=df['prob_unlike_err'], 
                             color=base_color, marker='x', markersize=6, linestyle='--', label=f"{label} (unlike)")


    axs1[0].set_title("One-Body Density")
    axs1[0].set_ylabel(r"$\rho(r)$")
    axs1[0].set_xlabel(r"$r$")
    axs1[0].legend()
    
    axs1[1].set_title("One-Body Radial Probability")
    axs1[1].set_ylabel(r"$P(r)$")
    axs1[1].set_xlabel(r"$r$")
    axs1[1].legend()
    fig1.tight_layout()

    if has_pcorr:
        axs2[0].set_title("Pair Correlation Density")
        axs2[0].set_ylabel(r"$\rho_{pair}(r)$")
        axs2[0].set_xlabel(r"$r$")
        axs2[0].legend()
        
        axs2[1].set_title("Pair Correlation Radial Probability")
        axs2[1].set_ylabel(r"$P_{pair}(r)$")
        axs2[1].set_xlabel(r"$r$")
        axs2[1].legend()
        fig2.tight_layout()
    else:
        plt.close(fig2) 

    # --- save ---
    os.makedirs("figs_OBD", exist_ok=True)
    if has_pcorr:
        os.makedirs("figs_pcorr", exist_ok=True)
        
    if len(fnames) == 1:
        save_base = fnames[0][4:-4] # e.g. 20260819_234333
    else:
        save_base = custom_save_name.strip()
        if not save_base:
            save_base = "multiplot"
        md = {}
    
    fig1.savefig(f"figs_OBD/OBD_{save_base}.png", dpi=300, metadata=md)
    print(f"Saved OBD plot to figs_OBD/OBD_{save_base}.png")
    if has_pcorr:
        fig2.savefig(f"figs_pcorr/pcorr_{save_base}.png", dpi=300, metadata=md)
        print(f"Saved Pair Correlation plot to figs_pcorr/pcorr_{save_base}.png")
        
    if talk:
        plt.show()

def on_select(listbox, entry_save_name, event=None):
    selection = listbox.curselection()
    if not selection:
        messagebox.showwarning("Warning", "Please select at least one file.")
        return
    
    fnames = [listbox.get(i) for i in selection]
    custom_save_name = entry_save_name.get()
    plot_files(fnames, custom_save_name)

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    matplotlib.rcParams.update({"axes.grid": True, "font.size": 13})

    if len(sys.argv) > 1 and sys.argv[1] == "all":
        fdir = "./logs_OBD"
        if os.path.exists(fdir):
            for file in sorted(os.listdir(fdir)):
                if file.endswith(".csv"):
                    plot_files([file], "", talk=False)
                    plt.close()
            print("Done!")
        else:
            print("Directory not found!")
        return

    root = tk.Tk()
    root.title("VMC Density & Pair Correlation Viewer")
    root.geometry("450x550")

    label = tk.Label(root, text="Select .csv file(s) from logs_OBD:\n(Use Ctrl/Shift for multiple)", font=("Arial", 11))
    label.pack(pady=10)

    frame = tk.Frame(root)
    frame.pack(padx=20, pady=5, fill=tk.BOTH, expand=True)

    scrollbar = tk.Scrollbar(frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    listbox = tk.Listbox(frame, yscrollcommand=scrollbar.set, font=("Courier", 10), selectmode=tk.EXTENDED)
    listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.config(command=listbox.yview)

    fdir = "./logs_OBD"
    if os.path.exists(fdir):
        for file in sorted(os.listdir(fdir)):
            if file.endswith(".csv"):
                listbox.insert(tk.END, file)
    else:
        listbox.insert(tk.END, "logs_OBD directory not found!")

    save_label = tk.Label(root, text="Save PNG as (for multi-select only):", font=("Arial", 10))
    save_label.pack(pady=(10, 0))
    
    save_name_var = tk.StringVar()
    save_entry = tk.Entry(root, textvariable=save_name_var, width=30)
    save_entry.pack(pady=5)

    listbox.bind('<Double-1>', lambda event: on_select(listbox, save_entry, event))

    plot_button = tk.Button(root, text="Plot Data", font=("Arial", 12, "bold"), 
                            bg="#2196F3", fg="white", 
                            command=lambda: on_select(listbox, save_entry))
    plot_button.pack(pady=15)

    root.mainloop()

if __name__ == "__main__":
    main()