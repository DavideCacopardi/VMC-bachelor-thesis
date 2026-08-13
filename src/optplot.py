import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
import tkinter as tk
from tkinter import messagebox

def readmetadata(fname):
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
    meta_dict = {"Description": lines}
    return meta_dict

def plot3D(x,
           y,
           energy,
           err,
           ax_sx,
           ax_dx,
           plot_surface=True,
           set_norm=True,
           norm_sx = None,
           norm_dx = None,
           s=30,
           c_min=30,
           label_min = "min",
           marker_min = '.',
           cmap = "viridis"):
    
    min_idx = np.argmin(energy)

    if set_norm:
        norm_sx = mcolors.PowerNorm(gamma=0.5, vmin=np.min(energy), vmax=np.max(energy))
    if plot_surface:
        plot_obj = ax_sx.plot_trisurf(x, y, energy, cmap=cmap, norm=norm_sx, edgecolor='none', alpha=0.85)
    else:
        plot_obj = ax_sx.scatter(x, y, energy, c=energy, cmap=cmap, norm=norm_sx, s=s)
    ax_sx.errorbar(x, y, energy, zerr=err, 
                    fmt='none', ecolor='black', alpha=0.3, lw=1.0, zorder=4)
    ax_sx.scatter(x[min_idx], y[min_idx], energy[min_idx], color=c_min, s=120, zorder=10, label=label_min, marker=marker_min)
    ax_sx.set_title(r"Energy")
    ax_sx.set_zlabel(r"$E$")
    ax_sx.set_xlabel(f"p[0]")
    ax_sx.set_ylabel(f"p[1]")
    # plt.colorbar(plot_obj, ax=ax, shrink=0.5)

    if set_norm:
        norm_dx = mcolors.PowerNorm(gamma=0.5, vmin=np.min(err**2), vmax=np.max(err**2))
    if plot_surface:
        plot_obj_var = ax_dx.plot_trisurf(x, y, err**2, cmap=cmap, norm=norm_dx, edgecolor='none', alpha=0.85)
    else:
        plot_obj_var = ax_dx.scatter(x, y, err**2, c=err**2, cmap=cmap, norm=norm_dx, s=s)
    ax_dx.scatter(x[min_idx], y[min_idx], err[min_idx]**2, color=c_min, s=120, zorder=10, label=label_min, marker=marker_min)
    ax_dx.set_title(r"Variance")
    ax_dx.set_zlabel(r"Var $E$")
    ax_dx.set_xlabel(f"p[0]")
    ax_dx.set_ylabel(f"p[1]")
    # plt.colorbar(plot_obj_var, ax=ax, shrink=0.5)
    return norm_sx, norm_dx


def plot4D(x, y, z, c, var, ax_sx, ax_dx):
    cmap = "viridis"
    s = 120

    min_idx = np.argmin(c)
    norm_sx = mcolors.PowerNorm(gamma=0.5, vmin=np.min(c), vmax=np.max(c))
    ax_sx.scatter(x, y, z, c=c, cmap=cmap, norm=norm_sx, s=s)
    ax_sx.scatter(x[min_idx], y[min_idx], z[min_idx], c=c[min_idx], s=120, zorder=10, marker='*')
    ax_sx.set_title(r"Energy")
    ax_sx.set_xlabel(f"p[0]")
    ax_sx.set_ylabel(f"p[1]")
    ax_sx.set_zlabel(f"p[2]")

    min_idx = np.argmin(var)
    norm_dx = mcolors.PowerNorm(gamma=0.5, vmin=np.min(var), vmax=np.max(var))
    ax_dx.scatter(x, y, z, c=var, cmap=cmap, norm=norm_dx, s=s)
    ax_dx.scatter(x[min_idx], y[min_idx], z[min_idx], c=var[min_idx], s=120, zorder=10, marker='*')
    ax_dx.set_title(r"Variance")
    ax_dx.set_xlabel(f"p[0]")
    ax_dx.set_ylabel(f"p[1]")
    ax_dx.set_zlabel(f"p[2]")


def rowSubplots2D(x, energy, err, fname, fig, gs, parNo = 0, row = 0):
    ax = fig.add_subplot(gs[row, 0])
    ax.errorbar(x, energy, err, color='magenta', ecolor='black',
                marker='.', markersize=8, linestyle='-', mfc="tab:blue", mec="tab:blue")
    ax.set_title(r"Energy")
    ax.set_ylabel(r"$E$")
    ax.set_xlabel(f"p[{parNo}]")
    ax.text(0.99, 0, fname[4:len(fname)-4], transform=ax.transAxes, fontsize=10, ha="right", va="bottom")

    ax = fig.add_subplot(gs[row, 1])
    ax.errorbar(x, err**2, color='magenta', ecolor='black',
                marker='.', markersize=8, linestyle='-', mfc="tab:blue", mec="tab:blue")
    ax.set_title(r"Variance")
    ax.set_ylabel(r"Var $E$")
    ax.set_xlabel(f"p[{parNo}]")
    

def plot_file(fname, talk = True):
    fdir = "./logs_opt"
    totfname = f"{fdir}/{fname}"
    
    print(f"Loading and plotting: {totfname}")
    try:
        with open(totfname) as file:
            line = file.readline()
            nPar = 0
            idx = line.find("p[")
            while idx != -1:
                line = line[idx + 2:]
                idx = line.find("p[")
                nPar += 1
        mesh = np.loadtxt(totfname, delimiter=',', usecols=np.arange(nPar + 3))
    except Exception as e:
        messagebox.showerror("Error", f"Could not load file {fname}:\n{e}")
        return

    # if nPar == 1:
    #     fig = plt.figure(figsize=(12, 6))
    #     gs = fig.add_gridspec(1, 2)
    #     rowSubplots2D(mesh[:,0], mesh[:,-3], mesh[:,-1], fname)
    # elif nPar == 2:
    #     p_x = 0
    #     p_y = 1
    #     fig = plt.figure(figsize=(12, 6))
    #     gs = fig.add_gridspec(1, 2)
    #     ax_sx = fig.add_subplot(gs[0, 0], projection='3d')
    #     ax_dx = fig.add_subplot(gs[0, 1], projection='3d')
    #     plot3D(mesh[:,p_x], mesh[:,p_y], mesh[:,-3], mesh[:,-1], ax_sx,ax_dx, False, set_norm=True, s=10, c_min="orange", label_min="min opt", marker_min='*')
    #     ax_sx.legend()
    #     ax_dx.legend()
    if nPar == 3:
        p_x = 0
        p_y = 1
        p_z = 2
        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(1, 2)
        ax_sx = fig.add_subplot(gs[0, 0], projection='3d')
        ax_dx = fig.add_subplot(gs[0, 1], projection='3d')
        plot4D(mesh[:,p_x], mesh[:,p_y], mesh[:,p_z], mesh[:,-3], mesh[:,-2], ax_sx,ax_dx)
        # ax_sx.legend()
        # ax_dx.legend()
        
    fig.tight_layout()
    md = readmetadata(f"logs/{fname[:-4]}.log")
    fig.savefig(f"figs_opt/plot_{fname[4:len(fname)-4]}.png", dpi=300, metadata=md)
    if talk:
        plt.show() 


def on_select(listbox, event=None):
    """Triggered when the user clicks the plot button or double-clicks a file."""
    selection = listbox.curselection()
    if not selection:
        messagebox.showwarning("Warning", "Please select a file from the list first.")
        return
    
    fname = listbox.get(selection[0])
    # Pass the state of the checkbox to the plotting function
    plot_file(fname)

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    matplotlib.rcParams.update({"axes.grid": True, "font.size": 15})

    if len(sys.argv) > 1 and sys.argv[1] == "all":
        fdir = "./logs_opt"
        if os.path.exists(fdir):
            for file in sorted(os.listdir(fdir)):
                if file.endswith(".csv"):
                    plot_file(file, talk=False)
            print("Done!")
        else:
            print("Directory not found!")
        return

    root = tk.Tk()
    root.title("VMC Optimization Viewer")
    root.geometry("400x600")

    label = tk.Label(root, text="Select a .csv file to plot:", font=("Arial", 12))
    label.pack(pady=10)

    frame = tk.Frame(root)
    frame.pack(padx=20, pady=5, fill=tk.BOTH, expand=True)

    scrollbar = tk.Scrollbar(frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    listbox = tk.Listbox(frame, yscrollcommand=scrollbar.set, font=("Courier", 10), selectmode=tk.SINGLE)
    listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.config(command=listbox.yview)

    listbox.bind('<Double-1>', lambda event: on_select(listbox, event))

    fdir = "./logs_opt"
    if os.path.exists(fdir):
        for file in sorted(os.listdir(fdir)):
            if file.endswith(".csv"):
                listbox.insert(tk.END, file)
    else:
        listbox.insert(tk.END, "Directory not found!")

    plot_button = tk.Button(root, text="Plot Selected File", font=("Arial", 12, "bold"), 
                            bg="#4CAF50", fg="white", 
                            command=lambda: on_select(listbox))
    plot_button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()