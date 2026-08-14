import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
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
    

def plot_file(fname, talk = True):
    fdir = "./logs_OBD"
    totfname = f"{fdir}/{fname}"
    
    print(f"Loading and plotting: {totfname}")
    try:
        # r, rho, drho, p, dp = np.loadtxt(totfname, delimiter=',')
        mat = np.loadtxt(totfname, delimiter=',')
        nFlavor = int((mat[0].size - 5) / 4)
        r = mat[:,0]
        rho = mat[:,1]
        drho = mat[:,2]
        p = mat[:,3+2*nFlavor]
        dp = mat[:,4+2*nFlavor]
        if nFlavor > 0:
            rhoFlav = np.zeros((r.size, nFlavor))
            drhoFlav = np.zeros((r.size, nFlavor))
            pFlav = np.zeros((r.size, nFlavor))
            dpFlav = np.zeros((r.size, nFlavor))
            for i in range(nFlavor):
                rhoFlav[:,i] = mat[:,3+2*i]
                drhoFlav[:,i] = mat[:,4+2*i]
                pFlav[:,i] = mat[:,5+2*nFlavor+2*i]
                dpFlav[:,i] = mat[:,6+2*nFlavor+2*i]
    except Exception as e:
        messagebox.showerror("Error", f"Could not load {fname}:\n{e}")
        return

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2)

    ax = fig.add_subplot(gs[0, 0])
    ax.errorbar(r, y=rho, yerr=drho, color='magenta', ecolor='black',
                marker='.', markersize=8, linestyle='-', mfc="darkblue", mec="darkblue", label="all")
    for i in range(nFlavor):
        flav = chr(ord('A') + i)
        ax.errorbar(r, y=rhoFlav[:,i], yerr=drhoFlav[:,i], ecolor='black',
                marker='.', markersize=4, linestyle='-', label=flav, alpha=0.5)
    ax.set_title(r"radially averaged density")
    ax.set_ylabel(r"$\rho(r)$")
    ax.set_xlabel(r"$r$")
    ax.legend()
    ax.text(0.99, 0, fname[4:len(fname)-4], transform=ax.transAxes, fontsize=10, ha="right", va="bottom")

    ax = fig.add_subplot(gs[0, 1])
    ax.errorbar(r, y=p, yerr=dp, color='magenta', ecolor='black',
                marker='.', markersize=8, linestyle='-', mfc="darkblue", mec="darkblue", label="all")
    for i in range(nFlavor):
        flav = chr(ord('A') + i)
        ax.errorbar(r, y=pFlav[:,i], yerr=dpFlav[:,i], ecolor='black',
                marker='*', markersize=4, linestyle='-', label=flav, alpha=0.5)
    ax.set_title(r"radial probability")
    ax.set_ylabel(r"$P(r)$")
    ax.set_xlabel(r"$r$")
    ax.legend()

    fig.tight_layout()
    md = readmetadata(f"logs/{fname[:-4]}.log")
    fig.savefig(f"figs_OBD/OBD_{fname[4:len(fname)-4]}.png", dpi=300, metadata=md)
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
        fdir = "./logs_OBD"
        if os.path.exists(fdir):
            for file in sorted(os.listdir(fdir)):
                if file.endswith(".csv"):
                    plot_file(file, talk=False)
            print("Done!")
        else:
            print("Directory not found!")
        return

    root = tk.Tk()
    root.title("VMC One Body Density Viewer")
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

    fdir = "./logs_OBD"
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