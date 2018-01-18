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
    H = np.sqrt(curvature_dp * (1 + z_val) ** 2 + matter_dp * (1 + z_val) ** 3 + lambda_dp)
    return 1. / H


def parallel(matter_dp, lambda_dp, zs_array):
    H0 = 70000  # km/s/Gpc assuming h = 0.70
    H0 = 100000  # in terms of h
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


def numerical_check(matter_dp, lambda_dp, zs_array, q0):
    H0 = 70000
    H0 = 100000
    c = 2.998e5
    test_dist = parallel(matter_dp, lambda_dp, zs_array)
    analytical_dist = -c / H0 * 2 * (1 / np.sqrt(zs_array + 1) - 1)  # om = 1, ol = 0
    # Liske_dist = -c / H0 / q0**2 / (1+zs_array) * (q0 * z + (q0 - 1) * (np.sqrt(1 + 2 * q0 * zs_array) - 1))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(zs_array, test_dist, label="Numerical")
    ax.plot(zs_array, analytical_dist, label="Analytical", linestyle="--")
    # ax.plot(zs_array, Liske_dist, label="Liske")  # Liske distance doesn't match
    ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(0.7, 0.8))
    ax.set_xlabel("$z$", fontsize=16)
    ax.set_ylabel("$R_0\chi$ (h$^{-1}$Gpc)", fontsize=16)
    ax.set_title("Numerical Versus Analytical Solution", fontsize=20)
    fig.savefig("analytical_check.pdf", bbox_inches="tight")
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
    ax.set_ylabel("$R_0\chi$ (h$^{-1}$Gpc)", fontsize=16)
    if curvature == 0:
        ax.set_title("Parallel Distance (Flat Space)", fontsize=20)
    elif curvature < 0:
        ax.set_title("Parallel Distance (Negative Curved Space)", fontsize=20)
    else:
        ax.set_title("Parallel Distance (Positive Curved Space)", fontsize=20)
    fig.savefig("parallel.pdf", bbox_inches="tight")
    plt.show()


def plot_perp_thet(z_contours, thetas_array, dist, curvature):
    colours = matplotlib.cm.rainbow(np.linspace(0, 1, z_contours.size))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("$\\theta$", fontsize=16)
    ax.set_ylabel("$R_0\chi$ (h$^{-1}$Gpc)", fontsize=16)
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
    fig.savefig("theta.pdf", bbox_inches="tight")


def plot_perp_phi(z_contours, phis_array, dists, curvature):
    colours = matplotlib.cm.rainbow(np.linspace(0, 1, z_contours.size))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("$\\phi$", fontsize=16)
    ax.set_ylabel("$R_0\chi\sin\\theta$ (h$^{-1}$Gpc)", fontsize=16)
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
    fig.savefig("phi.pdf", bbox_inches="tight")


def get_line_integrand(lambda_val, theta1, theta2, phi1, phi2):
    """Integrand for calculating line integral between two arbitrary points on a sphere.

    Inputs:
     lambda_val -- the current value of lambda, used to parametrise the line integral
     theta1, theta1 -- initial and final theta co-ordinates
     phi1, phi2 -- initial and final phi co-ordinates
    """
    theta = theta2 - theta1
    phi = phi2 - phi1
    theta_l = theta1 + lambda_val * theta
    integrand = np.sqrt(theta ** 2 + np.sin(theta_l) ** 2 * phi ** 2)  # from angular part of FRW metric,
    # or metric of a sphere, parametrised.
    return integrand


def calculate_line_integral(phi_val, radius, theta1, theta2, phi1):
    """Value of the line integral between two arbitrary points on a sphere

    Inputs:
     phi_val -- the final phi co-ordinate
     radius -- the parallel distance to a sphere of a given redshift
     theta1, theta2, phi1 -- initial and final co-ordinates of theta and phi
    """
    lambdas = np.linspace(0, 1, 101)
    integrands = get_line_integrand(lambdas, theta1, theta2, phi1, phi_val)
    integral = radius * sp.cumtrapz(integrands, x=lambdas, initial=0)
    return integral[-1]


def total_perp(matter_dp, lambda_dp, zs_array, z_contour, phis_array, theta1, theta2):
    """Calculates the total perpendicular distance in three ways.
    The ways are: approximation from pythagorean approximation using separate distance, analytical solution to arc
    length of great circle connecting the points (only used a check for the actual integration method), and the method
    of integrating the line element.

    Inputs:
     matter_dp, lambda_dp -- matter and dark energy density parameters
     zs_array -- array of redshifts for finding parallel distance
     z_contour -- the redshift giving the radius of the sphere
     phis_array -- array of phi values to find distances as a function of phi
    """
    phi1 = 0
    # Find approximate distance
    dist1 = perp_thet(matter_dp, lambda_dp, zs_array, z_contour, theta1)  # distance along theta to 1st pt.
    dist2 = perp_thet(matter_dp, lambda_dp, zs_array, z_contour, theta2)  # distance to theta co-ordinate of 2nd pt.
    theta_dist = dist2 - dist1
    phi_dist = perp_phi(matter_dp, lambda_dp, zs_array, z_contour, theta2, phis_array)  # array of phi distances, so
    # D_perp is a function of phi
    dist_perp = np.sqrt(theta_dist ** 2 + phi_dist ** 2)

    # # Find analytical value
    parallel_dists = parallel(matter_dp, lambda_dp, zs_array)  # parallel distances to z_max
    radius = parallel_dists[np.argmin(np.abs(zs_array - z_contour))]  # select only the distance corresponding to the
    # # radius of the sphere
    # analytical = radius * np.arccos(np.sin(theta1) * np.sin(theta2) * np.cos(phis_array) + np.cos(theta1)
    #                                 * np.cos(theta2))

    # Find numerical line integral distance
    numerical = vecCalculate_line_integral(phis_array, radius, theta1, theta2, phi1)

    difference = numerical - dist_perp
    norm_diff = difference/numerical
    return dist_perp, numerical, difference, norm_diff


def get_line_integrand_total(lambda_val, r1, r2, theta1, theta2, phi1, phi2):
    theta = theta2 - theta1
    phi = phi2 - phi1
    r = r2 - r1
    r_l = r1 + lambda_val * r
    integrand = np.sqrt(r**2 + r_l**2 * theta**2 + r_l**2 * np.sin(theta2)**2 * phi**2)
    return integrand


def calculate_line_integral_total(r_val, theta1, theta2, phi1, phi2, r1):
    lambdas = np.linspace(0, 1, 101)
    integrands = get_line_integrand_total(lambdas, theta1, r1, r_val, theta2, phi1, phi2)
    integral = sp.cumtrapz(integrands, x=lambdas, initial=0)
    return integral[-1]


def total_distance(matter_dp, lambda_dp, zs_array, z_contour, phi2, theta1, theta2):
    phi1 = 0
    # Find approximate distance
    dist1 = perp_thet(matter_dp, lambda_dp, zs_array, z_contour, theta1)
    dist2 = perp_thet(matter_dp, lambda_dp, zs_array, z_contour, theta2)
    theta_dist = dist2 - dist1
    phi_dist = perp_phi(matter_dp, lambda_dp, zs_array, z_contour, theta2, phi2)
    # D is a function of r
    r_dist = parallel(matter_dp, lambda_dp, zs_array)
    dist_total = np.sqrt(theta_dist**2 + phi_dist**2 + r_dist**2)

    # Find analytical value
    dists = parallel(matter_dp, lambda_dp, zs_array)
    radius = dists[np.argmin(np.abs(zs_array - z_contour))]
    numerical = vecCalculate_line_integral_total(zs_array, theta1, theta2, phi1, phi2, zs_array[0])

    difference = numerical - dist_total
    norm_diff = difference/numerical
    return dist_total, numerical, difference, norm_diff


def plot_total_perp(dists, numerical, diffs):
    fig = plt.figure()
    ax = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
    ax.set_ylabel("$D_\perp$ (h$^{-1}$Gpc)", fontsize=16)
    ax.set_title("Total Perpendicular Distance", fontsize=20)
    ax2 = plt.subplot2grid((4, 1), (3, 0))
    ax2.set_xlabel("$\phi$", fontsize=16)
    ax2.set_ylabel("Norm. Diff. (h$^{-1}$Gpc)")
    ax.tick_params(labelbottom=False)
    ax2.plot(phi_arr[:51], np.linspace(0, 0, 51), linestyle=":", color="k")
    colours = matplotlib.cm.rainbow(np.linspace(0, 1, theta_ends.size))
    for l, end in enumerate(theta_ends, start=0):
        ax.plot(phi_arr[:51], dists[l][:51], label=f"$\Delta\\theta$ = {int(np.ceil(theta_starts[l]/deg))}$^\circ$"
                                                   + f" - {int(theta_ends[l]/deg)}$^\circ$", color=colours[l])
        ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(0.05, 0.9))
    for l, end in enumerate(theta_ends, start=0):
        ax.plot(phi_arr[:51], numerical[l][:51], label="Numerical", linestyle="--", color=colours[l])

        ax2.plot(phi_arr[:51], diffs[l][:51], label="Difference", color=colours[l])

    plt.show()
    # fig.savefig("total_perp.pdf", bbox_inches="tight")


if __name__ == "__main__":
    vecGet_h_inv = np.vectorize(get_h_inv, excluded=['om', 'ol'])
    vecCalculate_line_integral = np.vectorize(calculate_line_integral, excluded=['radius', 'theta1', 'theta2', 'phi1'])
    vecCalculate_line_integral_total = np.vectorize(calculate_line_integral_total, excluded=['theta1', 'theta2', 'phi1', 'phi2', 'r1'])

    z_lo = 0.0
    z_hi = 15.0
    z_nstep = 101
    z_arr = np.linspace(z_lo, z_hi, z_nstep)

    om = 0.3
    ol = 0.7
    ok = 1.0 - om - ol

    phi_arr = np.linspace(0, 2 * np.pi, 101)
    theta_arr = np.linspace(0, 2 * np.pi, 101)

    # Calculate parallel distance for plotting
    dist_para = parallel(om, ol, z_arr)

    # Calculate theta perpendicular distance
    zs = np.linspace(z_lo + z_hi / 5, z_hi, int((z_nstep - 1) / 20))
    dist_thet = np.arange(zs.size * z_arr.size, dtype=np.float64).reshape(zs.size, z_arr.size)
    for i, z in enumerate(zs, start=0):
        dist_thet[i][:] = perp_thet(om, ol, z_arr, z, theta_arr)

    # Calculate phi perpendicular distance with two sample thetas
    deg = 0.01745  # 1 degree in radians.
    thetas = np.array([deg, 2 * deg])  # 1 and 2 degrees for phi perpendicular distance
    dist_phi = np.arange(zs.size * z_arr.size * thetas.size, dtype=np.float64).reshape(thetas.size, zs.size, z_arr.size)
    for i, angle in enumerate(thetas):
        for j, z in enumerate(zs, start=0):
            dist_phi[i][j] = perp_phi(om, ol, z_arr, z, angle, phi_arr)

    # Calculate the three total perpendicular distances
    z_radius = 1
    theta_starts = np.array([0, 0, 15 * deg, 15 * deg, 30 * deg, 30 * deg])
    theta_ends = np.array([5 * deg, 10 * deg, 20 * deg, 25 * deg, 35 * deg, 40 * deg])
    dist_perps = np.arange(theta_starts.size * phi_arr.size, dtype=np.float64).reshape(
        theta_starts.size, phi_arr.size)
    theory_perps = np.arange(theta_starts.size * phi_arr.size, dtype=np.float64).reshape(
        theta_starts.size, phi_arr.size)
    dist_diffs = np.arange(theta_starts.size * phi_arr.size, dtype=np.float64).reshape(
        theta_starts.size, phi_arr.size)
    for l, end in enumerate(theta_ends, start=0):
        dist_perps[l], theory_perps[l], _, dist_diffs[l] = total_perp(om, ol, z_arr, z_radius, phi_arr,
                                                                      theta_starts[l], theta_ends[l])

    # No need for these right now
    # numerical_check(om, ol, z_arr, 0.5)
    # plot_parallel(z_arr, dist_para, ok)
    # plot_perp_thet(zs, theta_arr, dist_thet, ok)
    # plot_perp_phi(zs, phi_arr, dist_phi, ok)
    plot_total_perp(dist_perps, theory_perps, dist_diffs)
