import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import messagebox
from mpl_toolkits.mplot3d import Axes3D

def readmetadata(fname):
    on = False
    lines = ""
    log_fname = f"logs/{fname[:-4]}.log"
    if not os.path.exists(log_fname):
        return {"Description": "No metadata found."}
        
    with open(log_fname) as file:
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
    

def plot_file(fname, talk=True):
    fdir = "./logs_particles"
    totfname = f"{fdir}/{fname}"
    
    print(f"Loading and plotting: {totfname}")
    try:
        df = pd.read_csv(totfname)
        df.columns = [c.replace('#', '').strip() for c in df.columns]
    except Exception as e:
        messagebox.showerror("Error", f"Could not load {fname}:\n{e}")
        return

    cols = df.columns.tolist()
    
    n_particles = sum('x0' in c for c in cols)
    has_flavor = any('flavor' in c for c in cols)
    n_snapshots = len(df)
    
    if any('x2' in c for c in cols):
        dim = 3
    elif any('x1' in c for c in cols):
        dim = 2
    else:
        dim = 1
        
    print(f"System: {dim}D | {n_particles} particles | {n_snapshots} steps")

    x_all, y_all, z_all, f_all, t_all = [], [], [], [], []
    
    for i in range(n_particles):
        if has_flavor:
            flavors = df[f'p{i} flavor'].astype(str).str.strip().values
        else:
            flavors = np.full(n_snapshots, 'A')
            
        x_all.extend(df[f'p{i} x0'].values)
        if dim >= 2:
            y_all.extend(df[f'p{i} x1'].values)
        else:
            y_all.extend(np.zeros(n_snapshots))
            
        if dim == 3:
            z_all.extend(df[f'p{i} x2'].values)
        else:
            z_all.extend(np.zeros(n_snapshots))
            
        f_all.extend(flavors)
        t_all.extend(np.arange(n_snapshots))

    p_df = pd.DataFrame({'x': x_all, 'y': y_all, 'z': z_all, 'flavor': f_all, 'time_idx': t_all})
    
    last_snapshot = p_df[p_df['time_idx'] == (n_snapshots - 1)]
    storico_df = p_df[p_df['time_idx'] < (n_snapshots - 1)]

    max_scatter_points = 30000 
    if len(storico_df) > max_scatter_points:
        sub_df = storico_df.sample(max_scatter_points)
    else:
        sub_df = storico_df.copy()
        
    sub_df = sub_df.sort_values('time_idx')

    color_map = {'A': mcolors.to_rgb('red'), 'B': mcolors.to_rgb('blue'), 
                 'C': mcolors.to_rgb('green'), 'D': mcolors.to_rgb('orange')}
    unique_flavors = np.unique(f_all)

    # pre-calculate fixed limits to prevent axis jitter during animation
    def get_limits(series):
        vmin, vmax = series.min(), series.max()
        margin = (vmax - vmin) * 0.05
        if margin == 0: margin = 1.0
        return vmin - margin, vmax + margin

    xlims = get_limits(p_df['x'])
    ylims = get_limits(p_df['y'])
    if dim == 3:
        zlims = get_limits(p_df['z'])

    # --- setup Figure ---
    fig = plt.figure(figsize=(8, 7))
    if dim == 2:
        ax = fig.add_subplot(111)
    elif dim == 3:
        ax = fig.add_subplot(111, projection='3d')
        
    alpha_min, alpha_max = 0.02, 0.6

    def draw_frame(t_curr):
        ax.clear()
        
        current_storico = sub_df[sub_df['time_idx'] < t_curr]
        current_last = p_df[p_df['time_idx'] == t_curr]
        
        for flav in unique_flavors:
            fl_data = current_storico[current_storico['flavor'] == flav]
            fl_last = current_last[current_last['flavor'] == flav]
            
            # Plot the trail up to the current frame
            if len(fl_data) > 0:
                t_norm = fl_data['time_idx'] / max(1, t_curr - 1)
                alphas = alpha_min + t_norm * (alpha_max - alpha_min)
                
                rgba_colors = np.zeros((len(fl_data), 4))
                rgba_colors[:, :3] = color_map.get(flav, mcolors.to_rgb('black'))
                rgba_colors[:, 3] = alphas
                
                if dim == 2:
                    ax.scatter(fl_data['x'], fl_data['y'], s=8, c=rgba_colors, edgecolors='none')
                elif dim == 3:
                    ax.scatter(fl_data['x'], fl_data['y'], fl_data['z'], 
                               s=10, c=rgba_colors, edgecolors='none')
            
            # Plot the 'leading' active points at the current frame
            if len(fl_last) > 0:
                lbl = f'Last step {flav}' if t_curr == n_snapshots - 1 else f'Step {t_curr} {flav}'
                if dim == 2:
                    ax.scatter(fl_last['x'], fl_last['y'], s=80, color=color_map.get(flav, 'black'),
                               edgecolors='yellow', linewidths=2.0, alpha=1.0, zorder=10, label=lbl)
                elif dim == 3:
                    ax.scatter(fl_last['x'], fl_last['y'], fl_last['z'], s=100, 
                               color=color_map.get(flav, 'black'), edgecolors='yellow', linewidths=2.0, 
                               alpha=1.0, zorder=10, label=lbl)

        # Restore labels and fixed limits
        ax.set_title(f"Particle evolution ({fname})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        
        if dim == 2:
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, alpha=0.3)
        elif dim == 3:
            ax.set_zlabel("Z")
            ax.set_zlim(zlims)
            
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='upper right')

    # Draw the final frame first to save a high-res static PNG
    draw_frame(n_snapshots - 1)
    fig.tight_layout()
    
    md = readmetadata(fname)
    os.makedirs("figs_particles", exist_ok=True)
    fig.savefig(f"figs_particles/Particles_{fname[4:len(fname)-4]}.png", dpi=300, metadata=md)
    print(f"Saved static final frame to figs_particles/Particles_{fname[4:len(fname)-4]}.png")
    
    if talk:
        fps = 20
        evo_duration = 5.0    # seconds of particle motion
        hold_duration = 12.0  # seconds holding the final fully-plotted frame
        
        n_evo_frames = int(fps * evo_duration)
        n_hold_frames = int(fps * hold_duration)
        
        time_steps = np.linspace(0, n_snapshots - 1, n_evo_frames, dtype=int)
        
        frames = list(time_steps) + [n_snapshots - 1] * n_hold_frames
        
        anim = FuncAnimation(fig, draw_frame, frames=frames, interval=1000/fps, blit=False, repeat=True)
        plt.show()

def on_select(listbox, event=None):
    selection = listbox.curselection()
    if not selection:
        messagebox.showwarning("Warning", "Please select a file from the list first.")
        return
    
    fname = listbox.get(selection[0])
    plot_file(fname)

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    matplotlib.rcParams.update({"axes.grid": True, "font.size": 12})

    fdir = "./logs_particles"
    
    if len(sys.argv) > 1 and sys.argv[1] == "all":
        if os.path.exists(fdir):
            for file in sorted(os.listdir(fdir)):
                if file.endswith(".csv"):
                    plot_file(file, talk=False)
            print("Done!")
        else:
            print("Directory not found!")
        return

    root = tk.Tk()
    root.title("VMC Particle Distribution Viewer")
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

    listbox.bind('<Double-1>', lambda event: on_select(listbox, event))

    if os.path.exists(fdir):
        for file in sorted(os.listdir(fdir)):
            if file.endswith(".csv"):
                listbox.insert(tk.END, file)
    else:
        listbox.insert(tk.END, f"Directory {fdir} not found!")

    plot_button = tk.Button(root, text="Plot Selected File", font=("Arial", 12, "bold"), 
                            bg="#4CAF50", fg="white", 
                            command=lambda: on_select(listbox))
    plot_button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()