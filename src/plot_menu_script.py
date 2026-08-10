import os
import sys
import numpy as np
import matplotlib.pyplot as plt
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
            if on and line.find("--") == -1 and line.find("[") == -1:
                lines += line
            idx = line.find("Seed")
            if idx != -1:
                on = False
                break
    meta_dict = {"Description": lines}
    return meta_dict

def plot_file(fname, plot_surface):
    fdir = "./parameter_mesh"
    totfname = f"{fdir}/{fname}"
    
    print(f"Loading and plotting: {totfname}")
    try:
        mesh = np.loadtxt(totfname, delimiter=',')
    except Exception as e:
        messagebox.showerror("Error", f"Could not load {fname}:\n{e}")
        return

    nPar = mesh.shape[1] - 2

    if nPar == 1:
        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(1, 2)

        ax = fig.add_subplot(gs[0, 0])
        ax.errorbar(mesh[:,0], y=mesh[:,-2], yerr=mesh[:,-1], color='magenta', ecolor='black',
                    marker='.', markersize=8, linestyle='-', mfc="tab:blue", mec="tab:blue")
        ax.set_title(r"Energy")
        ax.set_ylabel(r"$E$")
        ax.set_xlabel(f"p[0]")
        ax.text(0.99, 0, fname[4:len(fname)-4], transform=ax.transAxes, fontsize=10, ha="right", va="bottom")

        ax = fig.add_subplot(gs[0, 1])
        ax.errorbar(mesh[:,0], y=mesh[:,-1]**2, color='magenta', ecolor='black',
                    marker='.', markersize=8, linestyle='-', mfc="tab:blue", mec="tab:blue")
        ax.set_title(r"Variance")
        ax.set_ylabel(r"Var $E$")
        ax.set_xlabel(f"p[0]")

    elif nPar == 2:
        fig = plt.figure(figsize=(12, 6))
        min_idx = np.argmin(mesh[:,-2])

        ax = fig.add_subplot(121, projection='3d')
        if plot_surface:
            plot_obj = ax.plot_trisurf(mesh[:,0], mesh[:,1], mesh[:,-2], cmap='viridis', edgecolor='none', alpha=0.85)
        else:
            plot_obj = ax.scatter(mesh[:,0], mesh[:,1], mesh[:,-2], c=mesh[:,-2], cmap='viridis', s=30)
        ax.errorbar(mesh[:,p_x], mesh[:,p_y], mesh[:,-2], zerr=mesh[:,-1], 
                        fmt='none', ecolor='black', alpha=0.3, lw=1.0, zorder=4)
        ax.scatter(mesh[min_idx,0], mesh[min_idx,1], mesh[min_idx,-2], color='red', s=100, zorder=5)
        ax.set_title(r"Energy")
        ax.set_zlabel(r"$E$")
        ax.set_xlabel(f"p[0]")
        ax.set_ylabel(f"p[1]")
        plt.colorbar(plot_obj, ax=ax, shrink=0.5)

        ax = fig.add_subplot(122, projection='3d')
        if plot_surface:
            plot_obj_var = ax.plot_trisurf(mesh[:,0], mesh[:,1], mesh[:,-1]**2, cmap='viridis', edgecolor='none', alpha=0.85)
        else:
            plot_obj_var = ax.scatter(mesh[:,0], mesh[:,1], mesh[:,-1]**2, c=mesh[:,-1]**2, cmap='viridis', s=30)
        ax.scatter(mesh[min_idx,0], mesh[min_idx,1], mesh[min_idx,-1]**2, color='red', s=100, zorder=5)
        ax.set_title(r"Variance")
        ax.set_zlabel(r"Var $E$")
        ax.set_xlabel(f"p[0]")
        ax.set_ylabel(f"p[1]")
        plt.colorbar(plot_obj_var, ax=ax, shrink=0.5)
        
    else:
        min_idx = np.argmin(mesh[:,-2])
        okIdx_arr = []
        countRows = 0
        for par in range(nPar):
            mask = np.ones(mesh.shape[0], dtype=bool)
            for j in range(nPar):
                if j != par:
                    mask &= np.isclose(mesh[:, j], mesh[min_idx, j])
            
            okIdx = np.where(mask)[0]
            okIdx_arr.append(okIdx)
            if okIdx.size > 1:
                countRows += 1

        if countRows == 2:
            varied_params = [par for par, okIdx in enumerate(okIdx_arr) if okIdx.size > 1]
            p_x, p_y = varied_params[0], varied_params[1]

            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_subplot(121, projection='3d')
            if plot_surface:
                plot_obj = ax.plot_trisurf(mesh[:,p_x], mesh[:,p_y], mesh[:,-2], cmap='viridis', edgecolor='none', alpha=0.85)
            else:
                plot_obj = ax.scatter(mesh[:,p_x], mesh[:,p_y], mesh[:,-2], c=mesh[:,-2], cmap='viridis', s=30)
            ax.errorbar(mesh[:,p_x], mesh[:,p_y], mesh[:,-2], zerr=mesh[:,-1], 
                        fmt='none', ecolor='black', alpha=0.3, lw=1.0, zorder=4)
            ax.scatter(mesh[min_idx,p_x], mesh[min_idx,p_y], mesh[min_idx,-2], color='red', s=100, zorder=5)
            ax.set_title(r"Energy")
            ax.set_zlabel(r"$E$")
            ax.set_xlabel(f"p[{p_x}]")
            ax.set_ylabel(f"p[{p_y}]")
            plt.colorbar(plot_obj, ax=ax, shrink=0.5)

            ax = fig.add_subplot(122, projection='3d')
            if plot_surface:
                plot_obj_var = ax.plot_trisurf(mesh[:,p_x], mesh[:,p_y], mesh[:,-1]**2, cmap='viridis', edgecolor='none', alpha=0.85)
            else:
                plot_obj_var = ax.scatter(mesh[:,p_x], mesh[:,p_y], mesh[:,-1]**2, c=mesh[:,-1]**2, cmap='viridis', s=30)
            ax.scatter(mesh[min_idx,p_x], mesh[min_idx,p_y], mesh[min_idx,-1]**2, color='red', s=100, zorder=5)
            ax.set_title(r"Variance")
            ax.set_zlabel(r"Var $E$")
            ax.set_xlabel(f"p[{p_x}]")
            ax.set_ylabel(f"p[{p_y}]")
            plt.colorbar(plot_obj_var, ax=ax, shrink=0.5)
            
        else:
            fig = plt.figure(figsize=(12, 6 * countRows))
            gs = fig.add_gridspec(countRows, 2)
            it = 0
            for par in range(nPar):
                okIdx = okIdx_arr[par]
                if okIdx.size <= 1:
                    continue

                ax = fig.add_subplot(gs[it, 0])
                ax.errorbar(mesh[okIdx,par], y=mesh[okIdx,-2], yerr=mesh[okIdx,-1], color='magenta', ecolor='black',
                            marker='.', markersize=8, linestyle='-', mfc="tab:blue", mec="tab:blue")
                ax.set_title(r"Energy")
                ax.set_ylabel(r"$E$")
                ax.set_xlabel(f"p[{par}]")
                ax.text(0.99, 0, fname[4:len(fname)-4], transform=ax.transAxes, fontsize=10, ha="right", va="bottom")

                ax = fig.add_subplot(gs[it, 1])
                ax.errorbar(mesh[okIdx,par], y=mesh[okIdx,-1]**2, color='magenta', ecolor='black',
                            marker='.', markersize=8, linestyle='-', mfc="tab:blue", mec="tab:blue")
                ax.set_title(r"Variance")
                ax.set_ylabel(r"Var $E$")
                ax.set_xlabel(f"p[{par}]")
                
                it += 1

    fig.tight_layout()
    md = readmetadata(f"logs/{fname[:-4]}.log")
    fig.savefig(f"figs_mesh/plot_{fname[4:len(fname)-4]}.png", dpi=300, metadata=md)
    plt.show() 


def on_select(listbox, plot_surface_var, event=None):
    """Triggered when the user clicks the plot button or double-clicks a file."""
    selection = listbox.curselection()
    if not selection:
        messagebox.showwarning("Warning", "Please select a file from the list first.")
        return
    
    fname = listbox.get(selection[0])
    # Pass the state of the checkbox to the plotting function
    plot_file(fname, plot_surface_var.get())

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    root = tk.Tk()
    root.title("VMC Parameter Mesh Viewer")
    root.geometry("400x400")

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

    listbox.bind('<Double-1>', lambda event: on_select(listbox, plot_surface_var, event))

    fdir = "./parameter_mesh"
    if os.path.exists(fdir):
        for file in sorted(os.listdir(fdir)):
            if file.endswith(".csv"):
                listbox.insert(tk.END, file)
    else:
        listbox.insert(tk.END, "Directory not found!")

    plot_button = tk.Button(root, text="Plot Selected File", font=("Arial", 12, "bold"), 
                            bg="#4CAF50", fg="white", 
                            command=lambda: on_select(listbox, plot_surface_var))
    plot_button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()