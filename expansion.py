import numpy as np
import matplotlib, matplotlib.pyplot as plt
import scipy.integrate as sp


def get_h_inv(z_val, matter_dp, lambda_dp):
    """Integrand for calculating comoving distance.
    Assumes the lower bound on redshift is 0.

    Inputs:
     z -- upper redshift bound
     om -- matter density parameter
     ol -- dark energy density parameter
    """
    curvature_dp = 1.0 - matter_dp - lambda_dp
    h = np.sqrt(curvature_dp * (1 + z_val) ** 2 + matter_dp * (1 + z_val) ** 3 + lambda_dp)
    return 1. / h


def parallel(matter_dp, lambda_dp, zs_array):
    H0 = 70000  # km/s/Gpc
    c = 2.998e5  # km/s
    h_invs = vecGet_h_inv(zs_array, matter_dp, lambda_dp)
    curvature_dp = 1.0 - matter_dp - lambda_dp
    comoving_coord = sp.cumtrapz(h_invs, x=zs_array, initial=0)
    if curvature_dp == 0:
        dist = comoving_coord * c / H0
    elif curvature_dp > 0:
        R0 = c / (H0 * np.sqrt(curvature_dp))
        dist = R0 * np.sinh(comoving_coord / R0) * c / H0
    else:
        R0 = c / (H0 * np.sqrt(-curvature_dp))
        dist = R0 * np.sin(comoving_coord / R0) * c / H0
    return dist


def numerical_check(matter_dp, lambda_dp, zs_array):
    H0 = 70000
    c = 2.998e5
    test_dist = parallel(matter_dp, lambda_dp, zs_array)
    analytical_dist = -c / H0 * 2 * (1 / np.sqrt(zs_array + 1) - 1)  # om = 1, ol = 0
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(zs_array, test_dist, label="Numerical")
    ax.plot(zs_array, analytical_dist, label="Analytical")
    ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(0.7, 0.8))
    ax.set_xlabel("$z$", fontsize=16)
    ax.set_ylabel("$R_0\chi$ (Gpc)", fontsize=16)
    ax.set_title("Parallel Check", fontsize=20)
    # fig.savefig("analytical check.pdf", bbox_inches="tight")
    plt.show()


def perp_thet(matter_dp, lambda_dp, zs_array, z_contour, thetas_array):
    dists = parallel(matter_dp, lambda_dp, zs_array)
    para = dists[np.argmin(np.abs(zs_array - z_contour))]
    dist_final_theta = para * thetas_array
    return dist_final_theta


def perp_phi(matter_dp, lambda_dp, zs_array, z_contour, theta, phis_array):
    dists = parallel(matter_dp, lambda_dp, zs_array)
    para = dists[np.argmin(np.abs(zs_array - z_contour))]
    dist_final_phi = para * np.sin(theta) * phis_array
    return dist_final_phi


def plot_parallel(zs_array, dist, curvature):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(zs_array, dist, label=r"$\Omega_m=%0.2f$" % om)
    ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(0.7, 0.8))
    plt.figtext(0.85, 0.6, "$\Omega_\Lambda=%0.2f$" % ol, ha='right', va='bottom', weight='roman', size='large')
    ax.set_xlabel("$z$", fontsize=16)
    ax.set_ylabel("$R_0\chi$ (Gpc)", fontsize=16)
    if curvature == 0:
        ax.set_title("Parallel Distance (Flat Space)", fontsize=20)
    elif curvature < 0:
        ax.set_title("Parallel Distance (Negative Curved Space)", fontsize=20)
    else:
        ax.set_title("Parallel Distance (Positive Curved Space)", fontsize=20)
    # fig.savefig("parallel.pdf", bbox_inches="tight")
    plt.show()


def plot_perp_thet(z_contours, thetas_array, dist, curvature):
    colours = matplotlib.cm.rainbow(np.linspace(0, 1, z_contours.size))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("$\\theta$", fontsize=16)
    ax.set_ylabel("$R\chi$ (Gpc)", fontsize=16)
    for num, contour in enumerate(z_contours, start=0):
        ax.plot(thetas_array, dist[num], label=f"$z={contour}$", color=colours[num])
    ax.legend(loc="upper left", frameon=False)  # ,bbox_to_anchor=(0, 0.5))
    if curvature == 0:
        ax.set_title("$\\theta$ Perpendicular Distance (Flat Space)", fontsize=20)
    elif curvature < 0:
        ax.set_title("$\\theta$ Perpendicular Distance (Negative Curved Space)", fontsize=20)
    else:
        ax.set_title("$\\theta$ Perpendicular Distance (Positive Curved Space)", fontsize=20)
    plt.show()
    # fig.savefig("theta.pdf", bbox_inches="tight")


def plot_perp_phi(z_contours, phis_array, dists, curvature):
    colours = matplotlib.cm.rainbow(np.linspace(0, 1, z_contours.size))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("$\\phi$", fontsize=16)
    ax.set_ylabel("$R\chi\sin\\theta$ (Gpc)", fontsize=16)
    for num, contour in enumerate(z_contours, start=0):
        ax.plot(phis_array, dists[0][num], label=f"$z={contour}$", color=colours[num])
    ax.legend(loc="upper left", frameon=False)  # ,bbox_to_anchor=(0, 0.5))
    for num, contour in enumerate(z_contours, start=0):
        ax.plot(phis_array, dists[1][num], label=f"$z={contour}$", color=colours[num], linestyle="--")
    if curvature == 0:
        ax.set_title("$\\phi$ Perpendicular Distance (Flat Space)", fontsize=20)
    elif curvature < 0:
        ax.set_title("$\\phi$ Perpendicular Distance (Negative Curved Space)", fontsize=20)
    else:
        ax.set_title("$\\phi$ Perpendicular Distance (Positive Curved Space)", fontsize=20)
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
    ok = 1.0 - om - ol

    phi_arr = np.linspace(0, 2 * np.pi, 101)
    theta_arr = np.linspace(0, 2 * np.pi, 101)

    dist_para = parallel(om, ol, z_arr)

    zs = np.linspace(z_lo + z_hi / 5, z_hi, int((z_nstep - 1) / 20))
    dist_thet = np.arange(zs.size * z_arr.size, dtype=np.float64).reshape(zs.size, z_arr.size)
    for i, z in enumerate(zs, start=0):
        dist_thet[i][:] = perp_thet(om, ol, z_arr, z, theta_arr)

    thetas = np.array([0.01745, 2 * 0.01745])  # 1 and 2 degrees for phi perpendicular distance
    dist_phi = np.arange(zs.size * z_arr.size * thetas.size, dtype=np.float64).reshape(thetas.size, zs.size, z_arr.size)
    for i, angle in enumerate(thetas):
        for j, z in enumerate(zs, start=0):
            dist_phi[i][j] = perp_phi(om, ol, z_arr, z, angle, phi_arr)

    numerical_check(om, ol, z_arr)
    plot_parallel(z_arr, dist_para, ok)
    plot_perp_thet(zs, theta_arr, dist_thet, ok)
    plot_perp_phi(zs, phi_arr, dist_phi, ok)
