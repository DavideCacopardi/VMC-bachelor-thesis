import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tkinter as tk
from tkinter import messagebox
from datetime import datetime

def plot_blocking_files(fnames, custom_save_name, talk=True):
    fdir = "./logs_blocking"
    os.makedirs("figs_blocking", exist_ok=True)
    
    for fname in fnames:
        totfname = f"{fdir}/{fname}"
        print(f"Loading and plotting: {totfname}")
        
        try:
            col_names = ["thread_idx", "error", "selected_k"]
            df = pd.read_csv(totfname, comment='#', names=col_names, skipinitialspace=True)
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
        except Exception as e:
            messagebox.showerror("Error", f"Could not load {fname}:\n{e}")
            continue

        fig, ax = plt.subplots(figsize=(9, 6))
        fig.tight_layout()
        ax.ticklabel_format(style="sci", scilimits=(0,0), axis='y')
        ax.ticklabel_format(style="plain", axis='x')
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        cmap_colors = matplotlib.colormaps.get_cmap("tab10")
        cmap_colors = [cmap_colors.colors[i] for i in range(10)]
        
        grouped = df.groupby("thread_idx")
        
        optimal_k_values = []
        optimal_errors = []
        
        for thread_idx, group in grouped:
            block_indices = np.arange(len(group))
            error = group["error"].values
            
            k_val = int(group["selected_k"].iloc[0])
            optimal_k_values.append(k_val)
            optimal_errors.append(error[k_val])
            
            ax.plot(block_indices, error, marker='.', markersize=4, linestyle='-', c=cmap_colors[int(thread_idx) % 10],
                    alpha=0.4, label=f"Thread {int(thread_idx)}, K = {k_val}" if len(grouped) <= 10 else "")
            ax.errorbar(k_val, error[k_val], ls="none", marker='*', markersize=8, color=cmap_colors[int(thread_idx) % 10])
            
        for i in range(len(optimal_k_values)):
            ax.axvline(x=optimal_k_values[i], color=cmap_colors[i % 10], linestyle='--', linewidth=1, alpha=0.5)
            ax.axhline(y=optimal_errors[i], color=cmap_colors[i % 10], linestyle='--', linewidth=1, alpha=0.5)

        # most_frequent_k = max(set(optimal_k_values), key=optimal_k_values.count)
        # ax.axvline(x=most_frequent_k, color='red', linestyle='-', linewidth=2,
        #            label=r"Optimal Block Size ($\mathrm{Mode}"rf"(K)={most_frequent_k}$)")
        
        final_error = np.sqrt(np.sum(np.array(optimal_errors)**2)) / len(optimal_errors)
        ax.axhline(y=final_error, color='blue', linestyle='-', linewidth=2,
                   label="Final error")

        if len(grouped) <= 10:
            ax.legend(loc='lower right', fontsize=14)
        else:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend([handles[-1]], [labels[-1]], loc='lower right')

        ax.set_title(f"Automated Blocking")
        ax.set_xlabel(r"Blocking Transformation Step")
        ax.set_ylabel(r"Estimated Standard Error ($\sigma$)")
        fig.tight_layout()
        
        if custom_save_name:
            save_base = custom_save_name.strip()
        else:
            save_base = fname[:-4] 
            
        save_path = f"figs_blocking/blocking_{save_base}.png"
        fig.savefig(save_path, dpi=300)
        print(f"Saved Blocking plot to {save_path}")
        
        if talk:
            plt.show()
        else:
            plt.close(fig)

def on_select(listbox, entry_save_name, event=None):
    selection = listbox.curselection()
    if not selection:
        messagebox.showwarning("Warning", "Please select at least one file.")
        return
    
    fnames = [listbox.get(i) for i in selection]
    custom_save_name = entry_save_name.get()
    
    plot_blocking_files(fnames, custom_save_name)

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    plt.rcParams.update({"axes.grid": True, "font.size": 15})

    fdir = "./logs_blocking"
    
    if len(sys.argv) > 1 and sys.argv[1] == "all":
        if os.path.exists(fdir):
            for file in sorted(os.listdir(fdir)):
                if file.endswith(".csv"):
                    plot_blocking_files([file], "", talk=False)
            print("Done!")
        else:
            print(f"Directory {fdir} not found!")
        return

    root = tk.Tk()
    root.title("VMC Blocking Viewer")
    root.geometry("450x450")

    label = tk.Label(root, text=f"Select .csv file(s) from {fdir}:\n(Use Ctrl/Shift for multiple)", font=("Arial", 11))
    label.pack(pady=10)

    frame = tk.Frame(root)
    frame.pack(padx=20, pady=5, fill=tk.BOTH, expand=True)

    scrollbar = tk.Scrollbar(frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    listbox = tk.Listbox(frame, yscrollcommand=scrollbar.set, font=("Courier", 10), selectmode=tk.EXTENDED)
    listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.config(command=listbox.yview)

    if os.path.exists(fdir):
        for file in sorted(os.listdir(fdir)):
            if file.endswith(".csv"):
                listbox.insert(tk.END, file)
    else:
        listbox.insert(tk.END, f"{fdir} directory not found!")

    save_label = tk.Label(root, text="Save PNG prefix (optional):", font=("Arial", 10))
    save_label.pack(pady=(10, 0))
    
    save_name_var = tk.StringVar()
    save_entry = tk.Entry(root, textvariable=save_name_var, width=30)
    save_entry.pack(pady=5)

    listbox.bind('<Double-1>', lambda event: on_select(listbox, save_entry, event))

    plot_button = tk.Button(root, text="Plot Data", font=("Arial", 12, "bold"), 
                            bg="#4CAF50", fg="white", 
                            command=lambda: on_select(listbox, save_entry))
    plot_button.pack(pady=15)

    root.mainloop()

if __name__ == "__main__":
    main()