import numpy as np
import matplotlib, matplotlib.pyplot as plt
import scipy.integrate as sp

colours = ['C0', 'C1', 'C2', 'C3', 'C4', 'C9', 'C6', 'C7', 'C8', 'C5', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
col = [[0, 0.8, 1], [1, 0.85, 0], [0, 1, 0.1], [1, 0.6, 0.7], [0.8, 0, 1], [0, 1, 1]]


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
    # ax.set_title("Numerical Versus Analytical Solution", fontsize=20)
    # fig.savefig("analytical_check.pdf", bbox_inches="tight")
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
    ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(0.645, 0.8), fontsize=12)
    plt.figtext(0.85, 0.6, "$\Omega_\Lambda=%0.2f$" % ol, ha='right', va='bottom', weight='roman', fontsize=12)
    ax.set_xlabel("$z$", fontsize=16)
    ax.set_ylabel("$R_0\chi$ (h$^{-1}$Gpc)", fontsize=16)
    # if curvature == 0:
        # ax.set_title("Parallel Distance (Flat Space)", fontsize=20)
    # elif curvature < 0:
        # ax.set_title("Parallel Distance (Negative Curved Space)", fontsize=20)
    # else:
        # ax.set_title("Parallel Distance (Positive Curved Space)", fontsize=20)
    # fig.savefig("parallel.pdf", bbox_inches="tight")
    plt.show()


def plot_perp_thet(z_contours, thetas_array, dist, curvature):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("$\\theta$", fontsize=16)
    ax.set_ylabel("$R_0\chi$ (h$^{-1}$Gpc)", fontsize=16)
    for num, contour in enumerate(z_contours, start=0):
        ax.plot(thetas_array, dist[num], label=f"$z={contour}$", color=colours[num])
    ax.legend(loc="upper left", frameon=False, fontsize=12)  # ,bbox_to_anchor=(0, 0.5))
    # if curvature == 0:
    #     ax.set_title("$\\theta$ Perpendicular Distance (Flat Space)", fontsize=20)
    # elif curvature < 0:
    #     ax.set_title("$\\theta$ Perpendicular Distance (Negative Curved Space)", fontsize=20)
    # else:
    #     ax.set_title("$\\theta$ Perpendicular Distance (Positive Curved Space)", fontsize=20)
    plt.show()
    # fig.savefig("theta.pdf", bbox_inches="tight")


def plot_perp_phi(z_contours, phis_array, dists, curvature):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("$\\phi$", fontsize=16)
    ax.set_ylabel("$R_0\chi\sin\\theta$ (h$^{-1}$Gpc)", fontsize=16)
    for num, contour in enumerate(z_contours, start=0):
        ax.plot(phis_array, dists[0][num], label=f"$z={contour}$", color=colours[num])
    ax.legend(loc="upper left", frameon=False, fontsize=12)  # ,bbox_to_anchor=(0, 0.5))
    for num, contour in enumerate(z_contours, start=0):
        ax.plot(phis_array, dists[1][num], label=f"$z={contour}$", color=colours[num], linestyle="--")
    # if curvature == 0:
    #     ax.set_title("$\\phi$ Perpendicular Distance (Flat Space)", fontsize=20)
    # elif curvature < 0:
    #     ax.set_title("$\\phi$ Perpendicular Distance (Negative Curved Space)", fontsize=20)
    # else:
    #     ax.set_title("$\\phi$ Perpendicular Distance (Positive Curved Space)", fontsize=20)
    plt.show()
    # fig.savefig("phi.pdf", bbox_inches="tight")


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
    phi1 = 0.0
    # Find approximate distance
    dist1 = perp_thet(matter_dp, lambda_dp, zs_array, z_contour, theta1)  # distance along theta to 1st pt.
    dist2 = perp_thet(matter_dp, lambda_dp, zs_array, z_contour, theta2)  # distance to theta co-ordinate of 2nd pt.
    theta_dist = dist2 - dist1
    phi_dist = perp_phi(matter_dp, lambda_dp, zs_array, z_contour, theta2, phis_array)  # array of phi distances, so
    # D_perp is a function of phi
    angles_btw = np.arccos(np.sin(theta1) * np.sin(theta2) * np.cos(phis_array) + np.cos(theta1) * np.cos(theta2))
    dist_perp = np.sqrt(theta_dist ** 2 + phi_dist ** 2 - 2 * theta_dist * phi_dist * np.cos(angles_btw))
    # dist_perp = np.sqrt(theta_dist ** 2 + phi_dist ** 2)

    # # Find analytical value
    parallel_dists = parallel(matter_dp, lambda_dp, zs_array)  # parallel distances to z_max
    radius = parallel_dists[np.argmin(np.abs(zs_array - z_contour))]  # select only the distance corresponding to the
    # # radius of the sphere
    analytical = radius * np.arccos(np.sin(theta1) * np.sin(theta2) * np.cos(phis_array) + np.cos(theta1)
                                    * np.cos(theta2))

    # Find Liske's approximated distance
    liske_dist = radius * np.sqrt(2 - 2 * np.cos(angles_btw))

    difference = analytical - liske_dist
    norm_diff = difference/analytical
    return dist_perp, liske_dist, difference, norm_diff, analytical


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
    # ax.set_title("Total Perpendicular Distance", fontsize=20)
    ax2 = plt.subplot2grid((4, 1), (3, 0))
    ax2.set_xlabel("$\phi$", fontsize=16)
    ax2.set_ylabel("Norm. Diff. (h$^{-1}$Gpc)")
    ax.tick_params(labelbottom=False)
    ax2.plot(phi_arr[:501], np.linspace(0, 0, 501), linestyle=":", color="k")
    for l, end in enumerate(theta_ends, start=0):
        ax.plot(phi_arr[:501], dists[l][:501], label=f"$\Delta\\theta$ = {int(np.ceil(theta_starts[l]/deg))}$^\circ$"
                                                   + f" - {int(theta_ends[l]/deg)}$^\circ$", color=colours[l])
        ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(0.05, 0.9))
    for l, end in enumerate(theta_ends, start=0):
        ax.plot(phi_arr[:501], numerical[l][:501], label="Numerical", linestyle="--", color=colours[l])

        ax2.plot(phi_arr[:501], diffs[l][:501], label="Difference", color=colours[l])

    plt.show()
    # fig.savefig("total_perp.pdf", bbox_inches="tight")


def get_circle_integrand(theta_val, theta0, k):
    int = np.longdouble(np.cos(theta_val-theta0)**2)
    return np.longdouble(k / int)


def calculate_circle_integral(theta2, r):
    theta0 = theta2 / 2
    k = r * np.cos(theta0)
    theta_array = np.linspace(0, theta2, 1001)
    integrands = get_circle_integrand(theta_array, theta0, k)
    integral = sp.cumtrapz(integrands, x=theta_array, initial=0)
    return integral[-1]


def calculate_polar_integral(theta0, theta2, r):
    k = r * np.cos(theta0)
    theta_array = np.linspace(0, theta2, 1001)
    integrands = get_circle_integrand(theta_array, theta0, k)
    integral = sp.cumtrapz(integrands, x=theta_array, initial=0)
    return integral


def polar(r, theta_range):
    """Finds the chord, arc and geodesic distance of a circle."""
    # Chord length
    chord = 2 * r * np.sin(theta_range / 2)

    # Arc length
    arc = r * theta_range

    # Geodesic distance
    geo = vecCalculate_circle_integral(theta_range, r)
    # geo = 2 * r * np.cos(theta_range/2) * np.tan(theta_range - theta_range/2)

    norm_diff = (arc-geo)/geo

    return chord, arc, geo, norm_diff


def plot_polar(r):
    theta_range = np.linspace(np.longdouble(0), np.longdouble(np.pi), 1001)
    chord, arc, geo, nd = polar(r, theta_range)
    ax = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
    ax.set_ylabel("Length", fontsize=16)
    ax2 = plt.subplot2grid((4, 1), (3, 0))
    ax2.set_xlabel("$\\theta$", fontsize=16)
    ax2.set_ylabel("Norm. Diff.", fontsize=16)
    ax.tick_params(labelbottom=False)
    ax2.plot(theta_range, np.linspace(0, 0, 1001), linestyle="--", color="k")
    ax.plot(theta_range[:-5], geo[:-5], label="Geodesic distance")
    ax.plot(theta_range, chord, label="Chord length", linestyle='--')
    ax.plot(theta_range, arc, label="Arc length")
    ax2.plot(theta_range[:-5], nd[:-5])

    ax.legend(loc="lower right", frameon=False, bbox_to_anchor=(0.6, 0.7), fontsize=12)
    ax.set_ylabel("Length", fontsize=16)
    # ax.set_title("Comparison of Distance Measures in Polar Co-ordinates", fontsize=20)
    # fig.savefig("parallel.pdf", bbox_inches="tight")
    plt.show()


def geodesic_equations(s, r0, theta0):
    k = r0 * np.cos(theta0)
    k = r0
    r = np.sqrt(s**2 + k**2)
    theta = np.arctan(s/k) + theta0
    return r, theta


def rtheta(r0, theta, theta0):
    k = r0 * np.cos(theta0)
    k = r0
    return k / np.cos(theta-theta0)


def liske(x1, x2, alpha):
    return np.sqrt(x1**2 + x2**2 - 2*x1*x2*np.cos(alpha))


def roukema(chi1, chi2, alpha):
    # Convert to equatorial co-ordinates
    d1 = 0
    a1 = 0
    a2 = alpha
    d2 = 0

    x1 = chi1 * np.cos(d1) * np.cos(a1)
    x2 = chi2 * np.cos(d2) * np.cos(a2)
    y1 = chi1 * np.cos(d1) * np.sin(a1)
    y2 = chi2 * np.cos(d2) * np.sin(a2)
    z1 = chi1 * np.sin(d1)
    z2 = chi2 * np.sin(d2)

    inner_prod = (x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2
    return np.sqrt(inner_prod)


def liske_distances():
    chi1 = [2, 8]
    chi2 = [1, 2, 4, 2, 8, 10]
    alpha = np.linspace(0, np.pi/2, 501)

    # First comoving distance
    L11 = liske(chi1[0], chi2[0], alpha)
    L12 = liske(chi1[0], chi2[1], alpha)
    L13 = liske(chi1[0], chi2[2], alpha)
    R11 = roukema(chi1[0], chi2[0], alpha)
    R12 = roukema(chi1[0], chi2[1], alpha)
    R13 = roukema(chi1[0], chi2[2], alpha)

    # Second
    L21 = liske(chi1[1], chi2[3], alpha)
    L22 = liske(chi1[1], chi2[4], alpha)
    L23 = liske(chi1[1], chi2[5], alpha)
    R21 = roukema(chi1[1], chi2[3], alpha)
    R22 = roukema(chi1[1], chi2[4], alpha)
    R23 = roukema(chi1[1], chi2[5], alpha)

    plt.plot(alpha, L11, color=colours[0], label=f"$R_0\chi_1$ = {chi1[0]}, $R_0\chi_2$ = {chi2[0]}")
    plt.plot(alpha, L12, color=colours[1], label=f"$R_0\chi_1$ = {chi1[0]}, $R_0\chi_2$ = {chi2[1]}")
    plt.plot(alpha, L13, color=colours[2], label=f"$R_0\chi_1$ = {chi1[0]}, $R_0\chi_2$ = {chi2[2]}")
    plt.plot(alpha, R11, color=col[0], marker='o', markevery=50, linestyle='none', markersize=4)
    plt.plot(alpha, R12, color=col[1], marker='o', markevery=50, linestyle='none', markersize=4)
    plt.plot(alpha, R13, color=col[2], marker='o', markevery=50, linestyle='none', markersize=4)

    plt.plot(alpha, L21, color=colours[3], label=f"$R_0\chi_1$ = {chi1[1]}, $R_0\chi_2$ = {chi2[3]}")
    plt.plot(alpha, L22, color=colours[4], label=f"$R_0\chi_1$ = {chi1[1]}, $R_0\chi_2$ = {chi2[4]}")
    plt.plot(alpha, L23, color=colours[5], label=f"$R_0\chi_1$ = {chi1[1]}, $R_0\chi_2$ = {chi2[5]}")
    plt.plot(alpha, R21, color=col[3], marker='o', markevery=50, linestyle='none', markersize=4)
    plt.plot(alpha, R22, color=col[4], marker='o', markevery=50, linestyle='none', markersize=4)
    plt.plot(alpha, R23, color=col[5], marker='o', markevery=50, linestyle='none', markersize=4)
    plt.legend(loc="upper right", frameon=False, bbox_to_anchor=(0.5, 1))
    plt.xlabel("$\\alpha$", fontsize=16)
    plt.ylabel("$R_0\chi_{12}$ (h$^{-1}$Gpc)", fontsize=16)
    plt.show()


def dist_comp():
    #  Values necessary for numerical integration #
    R1 = 1
    theta1 = 0

    # theta0's to specify angle of line
    theta0 = np.array([np.deg2rad(0), np.deg2rad(-85), np.deg2rad(45), np.deg2rad(-45)])
    k = R1 * np.cos(theta0)

    # Integral limits
    theta2 = np.array([np.deg2rad(50), np.deg2rad(1), np.deg2rad(110), np.deg2rad(30)])

    s = np.arange(1001*4, dtype=np.float64).reshape(4, 1001)
    for m in [0, 1, 2, 3]:
        s[m] = calculate_polar_integral(theta0[m], theta2[m], R1)
        plt.plot(np.linspace(0, 1, 1001), s[m], label=f"$\\theta_0 = {int(np.rad2deg(theta0[m]))}^\circ$",
                 linewidth=1.8)

    # Values necessary for Liske's method
    R2 = np.array([np.sqrt(R1**2 + s[0][-1]**2),
                   np.sqrt(R1**2 + s[1][-1]**2 + 2*R1*s[1][-1]*np.cos(theta0[1]+np.pi/2)),
                   np.sqrt(R1**2 + s[2][-1]**2 - 2*R1*s[2][-1]*np.cos(theta0[2])),
                   np.sqrt(R1**2 + s[3][-1]**2 + 2*R1*s[3][-1]*np.cos(theta0[3]+np.pi/2))
                   ])

    theta = np.array([np.arccos(R1 / R2[0]),
                     np.arccos((R1 + s[1][-1] * np.cos(theta0[1]+np.pi/2)) / R2[1]),
                     np.arccos((R1 - s[2][-1] * np.cos(theta0[2])) / R2[2]),
                     np.arccos((R1 + s[3][-1] * np.cos(theta0[3]+np.pi/2)) / R2[3])
                      ])

    L = np.arange(1001*4, dtype=np.float64).reshape(4, 1001)
    R2 = np.arange(1001 * 4, dtype=np.float64).reshape(4, 1001)
    alpha = np.arange(1001 * 4, dtype=np.float64).reshape(4, 1001)
    col = [[0, 0.8, 1], [1, 0.85, 0], [0, 1, 0.1], [1, 0.6, 0.7]]
    for m in[0, 1, 2, 3]:
        alpha[m] = np.linspace(0, theta[m], 1001)
        R2[m] = rtheta(k[m], alpha[m], theta0[m])
        L[m] = liske(R1, R2[m], alpha[m])
        plt.plot(np.linspace(0, 1, 1001), L[m], linestyle='none', color=col[m], marker='o', markersize=4,
                 markevery=50) #, label=f"Liske $\\theta_0 = "
                                                                                    # f"{int(np.rad2deg(theta0[m]))}"
                                                                                    # f"^\circ$")
    plt.xlabel("s", fontsize=16)
    plt.ylabel("Distance", fontsize=16)
    plt.legend(loc="upper right", frameon=False, bbox_to_anchor=(0.5, 1), fontsize=12)
    plt.show()


# Geodesics on the surface of a sphere #
# def t_of_p(p, p0, A):
#     return np.tan(1 / (A * np.cos(p - p0)))
#
#
# def dtdp(p, p0, A):
#     return A * np.sin(p - p0) * np.sin(t_of_p(p, p0, A))**2
#
#
# def getint(pval, p0, A):
#     return np.sqrt(dtdp(pval, p0, A)**2 + np.sin(t_of_p(pval, p0, A))**2)
#
#
# def great_circle(t1, t2, p1, p2, R):
#     a = R**2 * (np.cos(t2)*np.sin(p1)*np.sin(t1) - np.cos(t1)*np.sin(p2)*np.sin(t2))
#     b = R**2 * (np.cos(t1)*np.cos(p2)*np.sin(t2) - np.cos(p1)*np.cos(t2)*np.sin(t1))
#     c = R**2 * (np.cos(p1)*np.sin(t1)*np.sin(p2)*np.sin(t2) - np.cos(p2)*np.sin(p1)*np.sin(t1)*np.sin(t2))
#     A = -np.sqrt(a**2 + b**2)/c
#     p0 = np.arccos(a/np.sqrt(a**2 + b**2))
#     p_arr = np.linspace(p1, p2, 151)
#     dss = getint(p_arr, p0, A)
#     s = sp.cumtrapz(dss, x=p_arr, initial=0)
#     return s
#

if __name__ == "__main__":
    vecGet_h_inv = np.vectorize(get_h_inv, excluded=['om', 'ol'])
    vecCalculate_line_integral = np.vectorize(calculate_line_integral, excluded=['radius', 'theta1', 'theta2', 'phi1'])
    vecCalculate_line_integral_total = np.vectorize(calculate_line_integral_total, excluded=['theta1', 'theta2', 'phi1',
                                                                                             'phi2', 'r1'])
    vecCalculate_circle_integral = np.vectorize(calculate_circle_integral, excluded=[''])
    # vecGetint = np.vectorize(getint, excluded=['p0', 'A'])

    z_lo = 0.0
    z_hi = 15.0
    z_nstep = 1001
    z_arr = np.linspace(z_lo, z_hi, z_nstep)

    om = 0.3
    ol = 0.7
    ok = 1.0 - om - ol

    phi_arr = np.linspace(0, 2 * np.pi, 1001)
    theta_arr = np.linspace(0, 2 * np.pi, 1001)

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
    theta_starts = np.array([0, 0, 15 * deg, 15 * deg, 30 * deg, 89.9 * deg])
    theta_ends = np.array([5 * deg, 10 * deg, 20 * deg, 25 * deg, 35 * deg, 90.1 * deg])
    dists = np.arange(theta_starts.size * phi_arr.size, dtype=np.float64).reshape(
        theta_starts.size, phi_arr.size)
    analyts = np.arange(theta_starts.size * phi_arr.size, dtype=np.float64).reshape(
        theta_starts.size, phi_arr.size)
    theory_perps = np.arange(theta_starts.size * phi_arr.size, dtype=np.float64).reshape(
        theta_starts.size, phi_arr.size)
    dist_diffs = np.arange(theta_starts.size * phi_arr.size, dtype=np.float64).reshape(
        theta_starts.size, phi_arr.size)
    for l, end in enumerate(theta_ends, start=0):
        dists[l], theory_perps[l], _, dist_diffs[l], analyts[l] = total_perp(om, ol, z_arr, z_radius, phi_arr,
                                                                             theta_starts[l], theta_ends[l])

    larr = np.arange(101) / 50. - 1.0
    theta0 = np.array([np.pi/4, np.pi/6, np.pi/8])  # theta2 = theta0/2
    divs = [4, 6, 8]
    th_arr = np.arange(101) / 100.

    for k in [1, 2, 3]:
        r, theta1 = geodesic_equations(larr, k, theta0[k-1])
        x = r * np.cos(theta1)
        y = r * np.sin(theta1)
        plt.plot(x, y, color=colours[k-1], label=f"k = {k}, $\\theta_0$ = $\pi$/{divs[k-1]}")
        plt.plot([0, k*np.cos(theta0[k-1])], [0, k*np.sin(theta0[k-1])], color=colours[k-1], linestyle=':')
        plt.plot(x[25:51], np.sqrt(k ** 2 - x[25:51] ** 2), color=colours[k - 1],
                 linestyle=':')
        plt.plot(x[51:-5], np.sqrt(k ** 2 - x[51:-5] ** 2), color=colours[k - 1],
                 linestyle=':')
    plt.ylabel("y", fontsize=16)
    plt.xlabel("x", fontsize=16)
    plt.axis('equal')
    plt.legend(loc="upper left", frameon=False, bbox_to_anchor=(0, 1), fontsize=12)
    # plt.show()

    # r_theta2 = rtheta(1, th_arr * theta0[0] * 2, theta0[2])
    # plt.plot(th_arr * theta0[0] * 2, r_theta2, color=colours[3])
    for c, k in enumerate([1, 1, 1]):
        r_theta = rtheta(k, th_arr*theta0[c]*2, theta0[c])
        plt.plot(th_arr*theta0[c]*2, r_theta, color=colours[c], label=f"$\\theta_0$ = $\pi$/{divs[c]}")
    plt.legend(loc="upper right", frameon=False, bbox_to_anchor=(0.5, 1), fontsize=12)
    plt.xlabel("$\\theta$", fontsize=16)
    plt.ylabel("$r(\\theta)$", fontsize=16)
    plt.figtext(0.466, 0.648, f"k = {k}", ha='right', va='bottom', weight='roman', fontsize=12)
    plt.show()

    ang_diam_dist = parallel(om, ol, z_arr[:350])/(1+z_arr[:350])
    arc = 0.15/ang_diam_dist
    chord = 2*np.arcsin(0.15/2/ang_diam_dist)
    nd = (arc-chord)/chord
    # plt.plot(z_arr[:350], arc)
    plt.plot(z_arr[:350], chord)
    # plt.plot(z_arr[:350], nd)
    plt.plot(np.linspace(0.1, 0.1, 2), np.linspace(0, 1.25, 2), color='k', linestyle='--')
    plt.ylim(0, 1.25)
    plt.ylabel("$\\theta$", fontsize=16)
    plt.xlabel("z", fontsize=16)
    plt.show()
    # No need for these right now
    # numerical_check(om, ol, z_arr, 0.5)
    # plot_parallel(z_arr, dist_para, ok)
    # plot_perp_thet(zs, theta_arr, dist_thet, ok)
    # plot_perp_phi(zs, phi_arr, dist_phi, ok)
    # plot_total_perp(analyts, theory_perps, dist_diffs)
    # plot_polar(1)
    # dist_comp()
    # liske_distances()