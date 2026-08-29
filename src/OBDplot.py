import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tkinter as tk
from tkinter import messagebox
import scipy as sp

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

def plot_files(fnames, custom_save_name, legend_labels = None, talk=True):
    fdir = "./logs_OBD"
    
    fig1, axs1 = plt.subplots(1, 2, figsize=(12, 6))
    axs1[0].ticklabel_format(style="sci", scilimits=(0,0))
    axs1[1].ticklabel_format(style="sci", scilimits=(0,0))
    fig2, axs2 = plt.subplots(1, 2, figsize=(12, 6))
    axs2[0].ticklabel_format(style="sci", scilimits=(0,0))
    axs2[1].ticklabel_format(style="sci", scilimits=(0,0))
    
    cmap_colors = matplotlib.colormaps.get_cmap("tab20c")
    cmap_colors = [cmap_colors.colors[i] for i in range(20)]
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
        if legend_labels is None or legend_labels[i] == '':
            # e.g. '20260819_234333'
            label = fname[4:-4]
        else:
            label = legend_labels[i]
        base_color = cmap_colors[(4 * i)%20]
        base_color_A = cmap_colors[(4 * i + 4)%20]
        base_color_B = cmap_colors[(4 * i + 8)%20]
        sec_color = cmap_colors[(4 * i + 2)%20]
        
        # FIGURE 1: One-Body Density
        if 'dens_tot' in df.columns:
            err_col = 'dens_err_' if 'dens_err_' in df.columns else df.columns[2]
            
            axs1[0].errorbar(r, df['dens_tot'], yerr=df[err_col], 
                             color=base_color, marker='.', markersize=6, linestyle='-', label=rf"{label} (tot)")
            
            if 'dens_A' in df.columns and len(fnames) == 1:
                err_col = 'dens_err_A' if 'dens_err_A' in df.columns else df.columns[4]
                axs1[0].errorbar(r, df['dens_A'], yerr=df[err_col], marker='.', markersize=4, color=base_color_A, linestyle='-', alpha=0.4, label='A')
            if 'dens_B' in df.columns and len(fnames) == 1:
                err_col = 'dens_err_B' if 'dens_err_B' in df.columns else df.columns[6]
                axs1[0].errorbar(r, df['dens_B'], yerr=df[err_col], marker='s', mfc="none", markersize=4, color=base_color_B, linestyle='--', alpha=0.6, label='B')
                
        if 'prob_tot' in df.columns:
            err_col_prob = 'prob_err_' if 'prob_err_' in df.columns else [c for c in df.columns if 'prob_err' in c][0]
            axs1[1].errorbar(r, df['prob_tot'], yerr=df[err_col_prob], 
                             color=base_color, marker='.', markersize=6, linestyle='-', label=rf"{label} (tot)")
                             
            if 'prob_A' in df.columns and len(fnames) == 1:
                err_col = 'prob_err_A' if 'prob_err_A' in df.columns else [c for c in df.columns if 'prob_err' in c][1]
                axs1[1].errorbar(r, df['prob_A'], yerr=df[err_col], marker='.', markersize=4, color=base_color_A, linestyle='-', alpha=0.4, label='A')
            if 'prob_B' in df.columns and len(fnames) == 1:
                err_col = 'prob_err_B' if 'prob_err_B' in df.columns else [c for c in df.columns if 'prob_err' in c][2]
                axs1[1].errorbar(r, df['prob_B'], yerr=df[err_col], marker='s', mfc="none", markersize=4, color=base_color_B, linestyle='--', alpha=0.6, label='B')

        # FIGURE 2: Pair Correlation (alike / unlike)
        if 'dens_alike' in df.columns:
            has_pcorr = True

            def gauss_obd(x, A, alpha):
                return A * np.exp(-x**2/alpha)
            
            try:
                popt, pcov = sp.optimize.curve_fit(gauss_obd, r, df['dens_tot'], p0=[df['dens_tot'][0], 4.0])
                alpha = popt[1]
                print(f"[{label}] OBD empirical gaussian width squared: {alpha:.4f}")
            except:
                print(f"[{label}] Fit failed, fallback to alpha=4.0")
                alpha = 4.0
            # alpha = 4
            
            # Note: in the convolution of two exp(-r^2/alpha), the exponent for r_rel becomes exp(-r_rel^2 / (2*alpha))
            
            r_mask = r < (2.5 * np.sqrt(abs(alpha))) # remove noise
            r_mask = r < (80 * np.sqrt(abs(alpha)))
            r_valid = r[r_mask]
            
        
            macro_envelope = np.exp(- (r_valid**2) / (2 * abs(alpha)))
            
            g_alike = df['dens_alike'][r_mask] / macro_envelope
            g_unlike = df['dens_unlike'][r_mask] / macro_envelope
            
            # Normalizziamo l'altezza in modo che l'asintoto sia 1.0
            # tail_alike = g_alike.iloc[-30:].mean() if len(g_alike) > 30 else 1.0
            # tail_unlike = g_unlike.iloc[-30:].mean() if len(g_unlike) > 30 else 1.0
            
            # g_alike = g_alike / tail_alike
            # g_unlike = g_unlike / tail_unlike

            axs2[0].errorbar(r_valid, g_alike,  
                             color=sec_color, marker='.', markersize=3, linestyle='-', alpha=0.8, label=rf"{label} (alike)")
            axs2[0].errorbar(r_valid, g_unlike,
                             color=base_color, marker='.', markersize=3, linestyle='-', alpha=0.8, label=rf"{label} (unlike)")
            
            axs2[1].errorbar(r, df['prob_alike'], yerr=df['prob_alike_err'], 
                             color=sec_color, marker='.', markersize=3, linestyle='-', alpha=0.8, label=rf"{label} (alike)")
            axs2[1].errorbar(r, df['prob_unlike'], yerr=df['prob_unlike_err'], 
                             color=base_color, marker='.', markersize=3, linestyle='-', alpha=0.8, label=rf"{label} (unlike)")


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
        
    if len(fnames) == 1 and custom_save_name == "":
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

def on_select(listbox, entry_save_name, entry_legend_name, event=None):
    selection = listbox.curselection()
    if not selection:
        messagebox.showwarning("Warning", "Please select at least one file.")
        return
    
    fnames = [listbox.get(i) for i in selection]
    custom_save_name = entry_save_name.get()
    
    entry_legend_name = entry_legend_name.get()
    if type(entry_legend_name) is str and len(entry_legend_name) > 0:
        legend_labels = entry_legend_name.split("€")
        print(legend_labels)
        if legend_labels is None:
            legend_labels = [entry_legend_name]
        while len(legend_labels) < len(fnames):
            legend_labels.append('')
        plot_files(fnames, custom_save_name, legend_labels)
    else:
        plot_files(fnames, custom_save_name)

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    matplotlib.rcParams.update({"axes.grid": True, "font.size": 15})

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
    root.geometry("500x600")

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

    save_label = tk.Label(root, text="Save PNG as:", font=("Arial", 10))
    save_label.pack(pady=(10, 0))
    
    save_name_var = tk.StringVar()
    save_entry = tk.Entry(root, textvariable=save_name_var, width=30)
    save_entry.pack(pady=5)

    legend_label = tk.Label(root, text="Legend labels sep. with '€':", font=("Arial", 10))
    legend_label.pack(pady=(10, 0))
    
    legend_name_var = tk.StringVar()
    legend_entry = tk.Entry(root, textvariable=legend_name_var, width=30)
    legend_entry.pack(pady=5)

    listbox.bind('<Double-1>', lambda event: on_select(listbox, save_entry, legend_entry, event))

    plot_button = tk.Button(root, text="Plot Data", font=("Arial", 12, "bold"), 
                            bg="#2196F3", fg="white", 
                            command=lambda: on_select(listbox, save_entry, legend_entry))
    plot_button.pack(pady=15)

    root.mainloop()

if __name__ == "__main__":
    main()