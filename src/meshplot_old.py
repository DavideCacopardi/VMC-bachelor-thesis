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
           cmap = "viridis",
           plot_opt = False):
    
    min_idx = np.argmin(energy)
    if not plot_opt:
        print(f"global minimum: ({x[min_idx]}, {y[min_idx]}) -> {energy[min_idx]} +- {err[min_idx]}")

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
    ax_sx.set_zlabel(r"$E$", c="brown", zorder=10)
    # ax_sx.set_zlim(0, 0.22)
    ax_sx.set_xlabel(f"p[0]", c="brown", zorder=10)
    ax_sx.set_ylabel(f"p[1]", c="brown", zorder=10)
    # plt.colorbar(plot_obj, ax=ax, shrink=0.5)

    if set_norm:
        norm_dx = mcolors.PowerNorm(gamma=0.5, vmin=np.min(err**2), vmax=np.max(err**2))
    if plot_surface:
        plot_obj_var = ax_dx.plot_trisurf(x, y, err**2, cmap=cmap, norm=norm_dx, edgecolor='none', alpha=0.85)
    else:
        plot_obj_var = ax_dx.scatter(x, y, err**2, c=err**2, cmap=cmap, norm=norm_dx, s=s)
    ax_dx.scatter(x[min_idx], y[min_idx], err[min_idx]**2, color=c_min, s=120, zorder=10, label=label_min, marker=marker_min)
    ax_dx.set_title(r"Variance")
    ax_dx.set_zlabel(r"Var $E$", c="brown", zorder=10)
    ax_dx.set_xlabel(f"p[0]", c="brown", zorder=10)
    ax_dx.set_ylabel(f"p[1]", c="brown", zorder=10)
    # plt.colorbar(plot_obj_var, ax=ax, shrink=0.5)
    return norm_sx, norm_dx

def rowSubplots2D(x,
                  energy,
                  err,
                  ax_sx,
                  ax_dx,
                  parNo = 0,
                  c_min=30,
                  label_min = "min",
                  marker_min = '.',
                  plot_opt = False,
                  ls = '-'):
    min_idx = np.argmin(energy)
    if not plot_opt:
        print(f"global minimum: ({x[min_idx]}) -> {energy[min_idx]} +- {err[min_idx]}")

    ax_sx.errorbar(x, energy, err, ecolor='black',
                marker='.', markersize=8, linestyle=ls)
    ax_sx.errorbar(x[min_idx], energy[min_idx], err[min_idx], color=c_min, ecolor='black',
                marker=marker_min, markersize=8, label = label_min)
    ax_sx.set_title(r"Energy")
    ax_sx.set_ylabel(r"$E$")
    ax_sx.set_xlabel(f"p[{parNo}]")

    ax_dx.errorbar(x, err**2, ecolor='black',
                marker='.', markersize=8, linestyle=ls)
    ax_dx.errorbar(x[min_idx], err[min_idx]**2, color=c_min, ecolor='black',
                marker=marker_min, markersize=8, label = label_min)
    ax_dx.set_title(r"Variance")
    ax_dx.set_ylabel(r"Var $E$")
    ax_dx.set_xlabel(f"p[{parNo}]")


def plot_slices_3par(mesh, fname):
    p0_vals = np.unique(mesh[:, 0])
    n_slices = len(p0_vals)
    
    # global minimum
    min_idx = np.argmin(mesh[:, -2])
    best_p0, best_p1, best_p2 = mesh[min_idx, 0], mesh[min_idx, 1], mesh[min_idx, 2]
    print(f"global minimum: ({best_p0}, {best_p1}, {best_p2}) -> {mesh[min_idx,-2]} +- {mesh[min_idx,-1]}")
    
    fig, axs = plt.subplots(n_slices, 2, figsize=(14, 6 * n_slices), squeeze=False)
    
    # minima
    vmin_E, vmax_E = np.min(mesh[:, -2]), np.percentile(mesh[:, -2], 85) # outliers
    vmin_V, vmax_V = np.min(mesh[:, -1]**2), np.percentile(mesh[:, -1]**2, 85)

    for i, p0 in enumerate(p0_vals):
        mask = mesh[:, 0] == p0
        p1 = mesh[mask, 1]
        p2 = mesh[mask, 2]
        energy = mesh[mask, -2]
        variance = mesh[mask, -1]**2
        
        # Energy
        ax_E = axs[i, 0]
        # tricontourf
        cf_E = ax_E.tricontourf(p1, p2, energy, levels=40, cmap='viridis_r', vmin=vmin_E, vmax=vmax_E)
        ax_E.set_ylabel("p[2]")
        if p0 == best_p0:
            ax_E.scatter(best_p1, best_p2, color='red', marker='*', s=300, edgecolor='black', label="Global Min")
            ax_E.legend()
            if len(p0_vals) > 1:
                ax_E.set_title(f"Energy at p[0] = {p0:.3f} (MINIMUM)", fontweight='bold')
            else:
                ax_E.set_title(f"Energy at p[0] = {p0:.3f}")
        else:
            ax_E.set_title(f"Energy at p[0] = {p0:.3f}")
        cb = fig.colorbar(cf_E, ax=ax_E, fraction=0.046, pad=0.04)
        cb.formatter.set_scientific(True)
        cb.formatter.set_powerlimits((0, 0))
        cb.update_ticks()

        ax_V = axs[i, 1]
        cf_V = ax_V.tricontourf(p1, p2, variance, levels=40, cmap='plasma', vmin=vmin_V, vmax=vmax_V)
        ax_V.set_ylabel("p[2]")
        if p0 == best_p0:
            ax_V.scatter(best_p1, best_p2, color='cyan', marker='*', s=300, edgecolor='black')
            if len(p0_vals) > 1:
                ax_V.set_title(f"Variance at p[0] = {p0:.3f} (MINIMUM)", fontweight='bold')
            else:
                ax_V.set_title(f"Variance at p[0] = {p0:.3f}")
        else:
            ax_V.set_title(f"Variance at p[0] = {p0:.3f}")
        cb = fig.colorbar(cf_V, ax=ax_V, fraction=0.046, pad=0.04)
        cb.formatter.set_scientific(True)
        cb.formatter.set_powerlimits((0, 0))
        cb.update_ticks()

        if i == n_slices - 1:
            ax_E.set_xlabel("p[1]")
            ax_V.set_xlabel("p[1]")

    # fig.suptitle(f"4D Parameter Landscape: {fname}", fontsize=18, y=1.02)
    return fig


def plot_file(fname, plot_surface, force_slices, opt_ifAvailable, talk = True):
    fdir = "./parameter_mesh"
    totfname = f"{fdir}/{fname}"
    
    print(f"Loading and plotting: {totfname}")
    try:
        mesh = np.loadtxt(totfname, delimiter=',')
    except Exception as e:
        messagebox.showerror("Error", f"Could not load file {fname}:\n{e}")
        return
    
    if opt_ifAvailable:
        fdir_opt = "./logs_opt"
        totfname_opt = f"{fdir_opt}/{fname}"
        try:
            mesh_opt = np.loadtxt(totfname_opt, delimiter=',', usecols=range(0, mesh.shape[1] + 1))
            plot_opt = True
        except Exception as e:
            messagebox.showerror("Error", f"file_opt {fname} not found")
            plot_opt = False
    else:
        plot_opt = False

    nPar = mesh.shape[1] - 2

    countRows = 0
    if nPar > 2:
        min_idx = np.argmin(mesh[:,-2])
        okIdx_arr = []
        for par in range(nPar):
            mask = np.ones(mesh.shape[0], dtype=bool)
            for j in range(nPar):
                if j != par:
                    mask &= np.isclose(mesh[:, j], mesh[min_idx, j])
            
            okIdx = np.where(mask)[0]
            okIdx_arr.append(okIdx)
            if okIdx.size > 1:
                countRows += 1
    print(f"countRows = {countRows}")
    if not force_slices and (nPar == 2 or countRows == 2):
        if nPar == 2:
            p_x = 0
            p_y = 1
        else:
            varied_params = [par for par, okIdx in enumerate(okIdx_arr) if okIdx.size > 1]
            p_x, p_y = varied_params[0], varied_params[1]
        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(1, 2)
        ax_sx = fig.add_subplot(gs[0, 0], projection='3d')
        ax_dx = fig.add_subplot(gs[0, 1], projection='3d')
        formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        for ax in (ax_sx, ax_dx):
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.set_major_formatter(formatter)
        norm_sx, norm_dx = plot3D(mesh[:,p_x], mesh[:,p_y], mesh[:,-2], mesh[:,-1], ax_sx,ax_dx, plot_surface, c_min="red", label_min="min mesh")
        if plot_opt:
            plot3D(mesh_opt[:,p_x], mesh_opt[:,p_y], mesh_opt[:,-3], mesh_opt[:,-1], ax_sx,ax_dx, False, False, norm_sx, norm_dx, s=10, c_min="orange", label_min="min opt", marker_min='*', plot_opt=True)
        ax_sx.tick_params(axis='both', which='major', labelsize=9.5)
        ax_sx.tick_params(axis='z', which='major', labelsize=9.5, labelrotation=45)
        ax_dx.tick_params(axis='both', which='major', labelsize=9.5)
        ax_dx.tick_params(axis='z', which='major', labelsize=9.5, labelrotation=45)
        ax_sx.legend()
        ax_dx.legend()
    elif countRows == 3 or (force_slices and (nPar == 2 or countRows == 2)):
        fig = plot_slices_3par(mesh, fname)
    else:
        fig = plt.figure(figsize=(12, 6 * countRows))
        gs = fig.add_gridspec(countRows, 2)
        it = 0
        for par in range(nPar):
            okIdx = okIdx_arr[par]
            if okIdx.size <= 1:
                continue
            ax_sx = fig.add_subplot(gs[it, 0])
            ax_dx = fig.add_subplot(gs[it, 1])
            rowSubplots2D(mesh[okIdx,par], mesh[okIdx,-2], mesh[okIdx,-1], ax_sx, ax_dx, par, c_min='red', marker_min='.', label_min="min mesh")
            if plot_opt:
                rowSubplots2D(mesh_opt[okIdx,par], mesh_opt[okIdx,-3], mesh_opt[okIdx,-1], ax_sx, ax_dx, par, plot_opt=True, c_min='orange', marker_min='*', label_min="min opt", ls="none")
            ax_sx.text(0.99, 0, fname[4:len(fname)-4], transform=ax_sx.transAxes, fontsize=10, ha="right", va="bottom")
            ax_sx.legend()
            ax_dx.legend()
            it += 1

    fig.tight_layout(pad = 2)
    md = readmetadata(f"logs/{fname[:-4]}.log")
    fig.savefig(f"figs_mesh/plot_{fname[4:len(fname)-4]}.png", dpi=300, metadata=md)
    if talk:
        plt.show() 


def on_select(listbox, plot_surface_var, force_slices_var, plot_optimization_var, event=None):
    """Triggered when the user clicks the plot button or double-clicks a file."""
    selection = listbox.curselection()
    if not selection:
        messagebox.showwarning("Warning", "Please select a file from the list first.")
        return
    
    fname = listbox.get(selection[0])
    # Pass the state of the checkbox to the plotting function
    plot_file(fname, plot_surface_var.get(), force_slices_var.get(), plot_optimization_var.get())

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    matplotlib.rcParams.update({"axes.grid": True, "font.size": 15})

    if len(sys.argv) > 1 and sys.argv[1] == "all":
        fdir = "./parameter_mesh"
        if os.path.exists(fdir):
            for file in sorted(os.listdir(fdir)):
                if file.endswith(".csv"):
                    plot_file(file, plot_surface=True, force_slices=False, opt_ifAvailable=False, talk=False)
                    plt.close()
            print("Done!")
        else:
            print("Directory not found!")
        return

    root = tk.Tk()
    root.title("VMC Parameter Mesh Viewer")
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

    # checkbutton for 3D surfaces
    plot_surface_var = tk.BooleanVar(value=True) # Default is True (surface)
    surface_check = tk.Checkbutton(root, text="Plot 3D Surfaces (Trisurf)", variable=plot_surface_var, font=("Arial", 10))
    surface_check.pack(pady=5)

    # checkbutton for force_slices
    force_slices_var = tk.BooleanVar(value=False) # Default is False
    force_slices = tk.Checkbutton(root, text="Force slices when 2 parameters are optimized", variable=force_slices_var, font=("Arial", 10))
    force_slices.pack(pady=5)

    # checkbutton for optimization plot
    plot_optimization_var = tk.BooleanVar(value=False) # Default is False
    optimization_check = tk.Checkbutton(root, text="Plot Optimization", variable=plot_optimization_var, font=("Arial", 10))
    optimization_check.pack(pady=10)
    
    listbox.bind('<Double-1>', lambda event: on_select(listbox, plot_surface_var, force_slices_var, plot_optimization_var, event))

    fdir = "./parameter_mesh"
    if os.path.exists(fdir):
        for file in sorted(os.listdir(fdir)):
            if file.endswith(".csv"):
                listbox.insert(tk.END, file)
    else:
        listbox.insert(tk.END, "Directory not found!")

    plot_button = tk.Button(root, text="Plot Selected File", font=("Arial", 12, "bold"), 
                            bg="#4CAF50", fg="white", 
                            command=lambda: on_select(listbox, plot_surface_var,  force_slices_var, plot_optimization_var))
    plot_button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()