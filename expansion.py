import numpy as np
import matplotlib, matplotlib.pyplot as plt
import scipy.integrate as sp


def get_h_inv(z, om, ol):
    """Integrand for calculating comoving distance.
    Assumes the lower bound on redshift is 0.

    Inputs:
     z -- upper redshift bound
     om -- matter density parameter
     ol -- dark energy density parameter
    """
    ok = 1.0 - om - ol
    h = np.sqrt(ok * (1 + z) ** 2 + om * (1 + z) ** 3 + ol)
    return 1. / h


def parallel(om, ol, z_arr):
    H0 = 70000  # km/s/Gpc
    c = 2.998e5  # km/s
    h_invs = vecGet_h_inv(z_arr, om, ol)
    ok = 1.0 - om - ol
    comoving = sp.cumtrapz(h_invs, x=z_arr, initial=0) * c / H0
    if ok == 0:
        dists = comoving / (1 + z_arr)
    elif ok > 0:
        R0 = c / (H0 * np.sqrt(ok))
        dists = R0 * np.sinh(comoving / R0) / (1 + z_arr)
    else:
        R0 = c / (H0 * np.sqrt(-ok))
        dists = R0 * np.sin(comoving / R0) / (1 + z_arr)
    return comoving, dists


def numerical_check(om, ol , z_arr):
    H0 = 70000
    c = 2.998e5
    test_com, test_dist = parallel(om, ol, z_arr)
    analytical_com = -c / H0 * 2 * (1 / np.sqrt(z_arr + 1) - 1)  # om = 1, ol = 0
    analytical_dist = (-c / H0 * 2 * (1 / np.sqrt(z_arr + 1) - 1)) / (z_arr + 1)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(z_arr, test_com, label="Numerical")
    ax.plot(z_arr, analytical_com, label="Analytical")
    ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(0.7, 0.8))
    ax.set_xlabel("$z$", fontsize=16)
    ax.set_ylabel("$R_0\chi$ (Gpc)", fontsize=16)
    ax.set_title("Calibration", fontsize=20)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(z_arr, test_dist, label="Numerical")
    ax.plot(z_arr, analytical_dist, label="Analytical")
    ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(0.7, 0.8))
    ax.set_xlabel("$z$", fontsize=16)
    ax.set_ylabel("$R_0\chi$ (Gpc)", fontsize=16)
    ax.set_title("Calibration", fontsize=20)
    # fig.savefig("comoving.pdf", bbox_inches="tight")
    plt.show()


def perp_thet(om, ol, z_arr, z, theta_arr):
    _, dist_par = parallel(om, ol, z_arr)
    dist_para = dist_par[np.argmin(np.abs(z_arr - z))]
    dist_thet = dist_para * theta_arr
    return dist_thet


def perp_phi(om, ol, z_arr, z, theta, phi_arr):
    _, dist_par = parallel(om, ol, z_arr)
    dist_para = dist_par[np.argmin(np.abs(z_arr - z))]
    dist_phi = dist_para * np.sin(theta) * phi_arr
    return dist_phi


def plot_parallel(z_arr, dist_para, comoving):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(z_arr, dist_para, label=r"$\Omega_m=%0.2f$" % om)
    ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(0.7, 0.9))
    plt.figtext(0.85, 0.8, "$\Omega_\Lambda=%0.2f$" % ol, ha='right', va='bottom', weight='roman', size='large')
    ax.set_xlabel("$z$", fontsize=16)
    ax.set_ylabel("$R\chi$ (Gpc)", fontsize=16)
    ax.set_title("Parallel Distance (Flat Space)", fontsize=20)
    # fig.savefig("comoving.pdf", bbox_inches="tight")
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(z_arr, comoving, label=r"$\Omega_m=%0.2f$" % om)
    ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(0.7, 0.8))
    plt.figtext(0.85, 0.7, "$\Omega_\Lambda=%0.2f$" % ol, ha='right', va='bottom', weight='roman', size='large')
    ax.set_xlabel("$z$", fontsize=16)
    ax.set_ylabel("$R_0\chi$ (Gpc)", fontsize=16)
    ax.set_title("Comoving Distance (Flat Space)", fontsize=20)
    # fig.savefig("parallel.pdf", bbox_inches="tight")
    plt.show()


def plot_perp_thet(zs, theta_arr, dist_thet):
    colours = matplotlib.cm.rainbow(np.linspace(0, 1, zs.size))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("$\\theta$", fontsize=16)
    ax.set_ylabel("$R\chi$ (Gpc)", fontsize=16)
    ax.set_title("$\\theta$ Perpendicular Distance (Flat Space)", fontsize=20)
    for i, z in enumerate(zs, start=0):
        ax.plot(theta_arr, dist_thet[i], label=f"$z={z}$", color=colours[i])
    ax.legend(loc="upper left", frameon=False)  # ,bbox_to_anchor=(0, 0.5))
    plt.show()
    # fig.savefig("theta.pdf", bbox_inches="tight")


def plot_perp_phi(zs, phi_arr, dist_phi):
    colours = matplotlib.cm.rainbow(np.linspace(0, 1, zs.size))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("$\\phi$", fontsize=16)
    ax.set_ylabel("$R\chi\sin\\theta$ (Gpc)", fontsize=16)
    ax.set_title("$\\phi$ Perpendicular Distance (Flat Space)", fontsize=20)
    for i, z in enumerate(zs, start=0):
        ax.plot(phi_arr, dist_phi[0][i], label=f"$z={z}$", color=colours[i])
    ax.legend(loc="upper left", frameon=False)  # ,bbox_to_anchor=(0, 0.5))
    for i, z in enumerate(zs, start=0):
        ax.plot(phi_arr, dist_phi[1][i], label=f"$z={z}$", color=colours[i], linestyle="--")
    plt.show()
    # fig.savefig("phi.pdf", bbox_inches="tight")


if __name__ == "__main__":
    vecGet_h_inv = np.vectorize(get_h_inv, excluded=['om', 'ol'])

    z_lo = 0.0
    z_hi = 15.0
    z_nstep = 101
    z_arr = np.linspace(z_lo, z_hi, z_nstep)

    om = 1.0
    ol = 0.0

    phi_arr = np.linspace(0, 2 * np.pi, 101)
    theta_arr = np.linspace(0, 2 * np.pi, 101)

    comoving, dist_para = parallel(om, ol, z_arr)

    zs = np.linspace(z_lo + z_hi / 5, z_hi, int((z_nstep - 1) / 20))
    dist_thet = np.arange(zs.size * z_arr.size, dtype=np.float64).reshape(zs.size, z_arr.size)
    for i, z in enumerate(zs, start=0):
        dist_thet[i][:] = perp_thet(om, ol, z_arr, z, theta_arr)

    thetas = np.array([0.01745, 2 * 0.01745])  # 1 and 2 degrees for phi perpendicular distance
    dist_phi = np.arange(zs.size * z_arr.size * thetas.size, dtype=np.float64).reshape(thetas.size, zs.size, z_arr.size)
    for i, angle in enumerate(thetas):
        for j, z in enumerate(zs, start=0):
            dist_phi[i][j] = perp_phi(om, ol, z_arr, z, angle, phi_arr)

    # numerical_check(om, ol, z_arr)
    plot_parallel(z_arr, dist_para, comoving)
    plot_perp_thet(zs, theta_arr, dist_thet)
    plot_perp_phi(zs, phi_arr, dist_phi)
