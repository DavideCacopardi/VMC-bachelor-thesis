import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tkinter as tk
from tkinter import messagebox

def readmetadata_info(fname):
    meta_dict = {"Description": ""}
    var_weight = 0.05
    N = None
    
    if not os.path.exists(fname):
        print(f"Warning: Log file {fname} non trovato.")
        return meta_dict, var_weight, N
        
    on = False
    lines = ""
    with open(fname) as file:
        for line in file:
            if line.find("Hamiltonian") != -1:
                on = True
            if on and line.find("--") == -1:
                lines += line
            if line.find("Called") != -1:
                on = False
                
            if "varOpt_weight" in line:
                try:
                    var_weight = float(line.split(":")[1].strip())
                except (IndexError, ValueError):
                    pass
            
            if "particles (N)" in line:
                try:
                    N = int(line.split(":")[1].strip())
                except (IndexError, ValueError):
                    pass
                    
    meta_dict["Description"] = lines
    return meta_dict, var_weight, N

def plot_single_file(fname, talk=True):
    fdir = "./logs_opt"
    totfname = f"{fdir}/{fname}"
    log_fname = f"logs/{fname[:-4]}.log"
    
    md, var_weight, N = readmetadata_info(log_fname)
    print(f"Loading {totfname} using varOpt_weight = {var_weight}")
    
    try:
        df = pd.read_csv(totfname)
        df.columns = [c.split()[-1] if '[' in c else c.strip().replace('#', '').strip() for c in df.columns]
    except Exception as e:
        messagebox.showerror("Error", f"Could not load file {fname}:\n{e}")
        return

    param_cols = [c for c in df.columns if 'p[' in c]
    if not param_cols and 'energy' in df.columns:
        energy_idx = list(df.columns).index('energy')
        param_cols = list(df.columns)[:energy_idx]

    n_params = len(param_cols)
    if 'energy' not in df.columns:
        messagebox.showerror("Error", "Column 'energy' not found in the CSV.")
        return

    if 'variance' in df.columns:
        variance = df['variance']
    elif 'error' in df.columns:
        variance = df['error']**2
    else:
        variance = np.zeros(len(df))

    loss = df['energy'] + var_weight * variance
    steps = df.index

    tot_plots = 3 + n_params
    cols = 3
    rows = int(np.ceil(tot_plots / cols))
    
    fig, axs = plt.subplots(rows, cols, figsize=(18, 4.5 * rows))
    axs = axs.flatten()

    axs[0].errorbar(steps, df['energy'], df['error'], color='blue', marker='.', markersize=4, linestyle='-', lw=1.5)
    axs[0].set_title("Energy Evolution")
    axs[0].set_xlabel("Optimization Step")
    axs[0].set_ylabel(r"$E$")

    axs[1].plot(steps, variance, color='orange', marker='.', markersize=4, linestyle='-', lw=1.5)
    axs[1].set_title("Variance")
    axs[1].set_xlabel("Optimization Step")
    axs[1].set_ylabel(r"Var($E$)")
    axs[1].set_yscale('log')

    axs[2].plot(steps, loss, color='red', marker='.', markersize=4, linestyle='-', lw=1.5)
    axs[2].set_title(f"Loss Function (E + {var_weight} * Var)")
    axs[2].set_xlabel("Optimization Step")
    axs[2].set_ylabel("Loss")

    colors = ['green', 'purple', 'brown', 'cyan', 'magenta', 'olive']
    for i, p_name in enumerate(param_cols):
        ax = axs[3 + i]
        ax.plot(steps, df[p_name], color=colors[i % len(colors)], marker='.', markersize=4, linestyle='-', lw=1.5)
        ax.set_title(f"Parameter: {p_name}")
        ax.set_xlabel("Optimization Step")
        ax.set_ylabel(p_name)

    for k in range(tot_plots):
        axs[k].grid(True, alpha=0.4)

    for j in range(tot_plots, len(axs)):
        axs[j].set_visible(False)

    fig.tight_layout()

    os.makedirs("figs_opt", exist_ok=True)
    save_path = f"figs_opt/opt_plot_{fname[4:len(fname)-4]}.png"
    fig.savefig(save_path, dpi=300, metadata=md)
    print(f"Plot saved to {save_path}")
    
    if talk:
        plt.show()

def plot_multiple_files(fnames, custom_save_name, legend_labels = None, talk=True):
    all_param_cols = []
    data_list = []
    
    for fname in fnames:
        totfname = f"./logs_opt/{fname}"
        log_fname = f"logs/{fname[:-4]}.log"
        
        md, var_weight, N = readmetadata_info(log_fname)
        
        try:
            df = pd.read_csv(totfname)
            df.columns = [c.split()[-1] if '[' in c else c.strip().replace('#', '').strip() for c in df.columns]
        except Exception as e:
            print(f"Skipping {fname} due to error: {e}")
            continue
            
        param_cols = [c for c in df.columns if 'p[' in c]
        if not param_cols and 'energy' in df.columns:
            energy_idx = list(df.columns).index('energy')
            param_cols = list(df.columns)[:energy_idx]
            
        for p in param_cols:
            if p not in all_param_cols:
                all_param_cols.append(p)
                
        data_list.append((fname, df, var_weight, N))

    if not data_list:
        return

    tot_plots = 3 + len(all_param_cols)
    cols = 3
    rows = int(np.ceil(tot_plots / cols))
    
    fig, axs = plt.subplots(rows, cols, figsize=(18, 4.5 * rows))
    axs = axs.flatten()
    
    # cmap_colors = plt.cm.tab10(np.linspace(0, 1, len(data_list)))
    cmap_colors = matplotlib.colormaps.get_cmap("tab10")
    cmap_colors = [cmap_colors.colors[i] for i in range(10)]

    for i, (fname, df, var_weight, N) in enumerate(data_list):
        if legend_labels is None or legend_labels[i] == '':
            # e.g. '20260819_234333'
            label = fname[4:-4]
        else:
            label = legend_labels[i]
        base_color = cmap_colors[i % 10]
        steps = df.index
        
        if 'variance' in df.columns:
            variance = df['variance']
        elif 'error' in df.columns:
            variance = df['error']**2
        else:
            variance = np.zeros(len(df))
            
        loss = df['energy'] + var_weight * variance
        
        axs[0].errorbar(steps, df['energy'], df['error'], label=rf"{label}", color=base_color, marker='.', markersize=4, linestyle='-', lw=1.5)
        axs[1].plot(steps, variance, label=rf"{label}", color=base_color, marker='.', markersize=4, linestyle='-', lw=1.5)
        axs[2].plot(steps, loss, label=rf"{label}", color=base_color, marker='.', markersize=4, linestyle='-', lw=1.5)
        
        for j, p_name in enumerate(all_param_cols):
            ax = axs[3 + j]
            if p_name in df.columns:
                ax.plot(steps, df[p_name], label=rf"{label}", color=base_color, marker='.', markersize=4, linestyle='-', lw=1.5)

    axs[0].set_title("Energy")
    axs[0].set_ylabel(r"$E$")
    axs[1].set_title("Variance")
    axs[1].set_ylabel(r"Var($E$)")
    axs[1].set_yscale('log')
    axs[2].set_title("Loss Function")
    axs[2].set_ylabel("Loss")

    for j, p_name in enumerate(all_param_cols):
        axs[3 + j].set_title(f"Parameter: {p_name}")
        axs[3 + j].set_ylabel(p_name)

    for k in range(tot_plots):
        axs[k].set_xlabel("Optimization Step")
        axs[k].grid(True, alpha=0.4)
        if k != 1:
            axs[k].ticklabel_format(style="sci", scilimits=(0,0))
        axs[k].legend(markerfirst = False, alignment="right", framealpha=0.4, borderaxespad=0.15, fontsize=15)

    for j in range(tot_plots, len(axs)):
        axs[j].set_visible(False)

    fig.tight_layout()

    custom_save_name = custom_save_name.strip()
    if custom_save_name:
        os.makedirs("figs_opt", exist_ok=True)
        if not custom_save_name.endswith('.png'):
            custom_save_name += '.png'
        save_path = f"figs_opt/{custom_save_name}"
        fig.savefig(save_path, dpi=300)
        print(f"Multi-plot saved to {save_path}")
    else:
        print("No output filename provided. Multi-plot not saved.")

    if talk:
        plt.show()

def on_select(listbox, entry_save_name, entry_legend_name, event=None):
    selection = listbox.curselection()
    if not selection:
        messagebox.showwarning("Warning", "Please select at least one file.")
        return
    
    if len(selection) == 1:
        fname = listbox.get(selection[0])
        plot_single_file(fname)
    else:
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
            plot_multiple_files(fnames, custom_save_name, legend_labels)
        else:
            plot_multiple_files(fnames, custom_save_name)

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    matplotlib.rcParams.update({"axes.grid": True, "font.size": 17})

    if len(sys.argv) > 1 and sys.argv[1] == "all":
        fdir = "./logs_opt"
        if os.path.exists(fdir):
            for file in sorted(os.listdir(fdir)):
                if file.endswith(".csv"):
                    plot_single_file(file, talk=False)
            print("Done!")
        else:
            print("Directory not found!")
        return

    root = tk.Tk()
    root.title("VMC Optimization Viewer")
    root.geometry("500x600")

    label = tk.Label(root, text="Select .csv file(s) from logs_opt:\n(Use Ctrl/Shift for multiple)", font=("Arial", 11))
    label.pack(pady=10)

    frame = tk.Frame(root)
    frame.pack(padx=20, pady=5, fill=tk.BOTH, expand=True)

    scrollbar = tk.Scrollbar(frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    listbox = tk.Listbox(frame, yscrollcommand=scrollbar.set, font=("Courier", 10), selectmode=tk.EXTENDED)
    listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.config(command=listbox.yview)

    fdir = "./logs_opt"
    if os.path.exists(fdir):
        for file in sorted(os.listdir(fdir)):
            if file.endswith(".csv"):
                listbox.insert(tk.END, file)
    else:
        listbox.insert(tk.END, "logs_opt directory not found!")

    save_label = tk.Label(root, text="Save PNG as (for multi-select only):", font=("Arial", 10))
    save_label.pack(pady=(10, 0))
    
    save_name_var = tk.StringVar()
    save_entry = tk.Entry(root, textvariable=save_name_var, width=30)
    save_entry.pack(pady=5)

    legend_label = tk.Label(root, text="Legend labels sep. with '€' (for multi-select only):", font=("Arial", 10))
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