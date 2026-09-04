import numpy as np
from scipy.linalg import eigh_tridiagonal

HBAR2_2M = 0.1
HBAR2_2MU = 0.2
SIGMA = 1
EPSILON = 1
V_CAP = 1.0e8

def E_cm(omega):
    """Exact CM ground-state energy — identical formula to the
    single-particle E0 in Table 4.1 of the thesis."""
    return 1.5 * omega * np.sqrt(2.0 * HBAR2_2M)

def U_LJ(r, alpha_tilde):
    temp = (SIGMA / r) ** 6
    pot = 4.0 * alpha_tilde * EPSILON * (temp ** 2 - temp)
    return np.minimum(pot, V_CAP)

def trap_width(omega):
    alpha = np.sqrt(2.0 * HBAR2_2M) / omega
    return np.sqrt(alpha)

def solve_relative_motion(omega, alpha_tilde, r_min=1e-3, r_max=50, n_points=100000):
    """
    Ground-state energy of
 
        -HBAR2_2MU * u''(r) + [0.25*omega**2*r**2 + U_AB(r)] * u(r) = E * u(r)
 
    with u(0) = u(r_max) = 0
    """
 
    r = np.linspace(r_min, r_max, n_points)
    h = r[1] - r[0]
 
    v = 0.25 * omega**2 * r**2 + (U_LJ(r, alpha_tilde) if alpha_tilde > 0 else 0.0)
 
    diag = 2.0 * HBAR2_2MU / h**2 + v
    offdiag = -HBAR2_2MU / h**2 * np.ones(n_points - 1)
 
    eigval, _ = eigh_tridiagonal(diag, offdiag, select='i', select_range=(0, 0))
    return eigval[0]

if __name__ == "__main__":
    omega1, omega2 = 0.111803, 0.027951
 
    print(f"{'omega':>10} {'alpha_tilde':>12} {'E_rel':>19} {'E_cm':>19} {'E_exact':>19}")
    for omega in (omega1, omega2):
        for alpha_tilde in (1.00, 0.40, 0.05, 0.00):
            e_rel = solve_relative_motion(omega, alpha_tilde)
            ecm = E_cm(omega)
            print(f"{omega:10.6f} {alpha_tilde:12.2f} {e_rel:.13e} {ecm:.13e} {e_rel+ecm:.13e}")