# Dexter Dysthe
# Dr. Johannes
# B9337
# 3 November 2021

import numpy as np
from scipy.stats import norm

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


# ----------------------------------- Functions for Calculating the Greeks ----------------------------------- #

def d1(S_t, K, sigma, r, time_till_mat):
    """
    :param S_t:           Current spot price
    :param K:             Strike price
    :param sigma:         Volatility
    :param r:             Current short rate
    :param time_till_mat: Length of time remaining in the option

    :return: Returns the parameter d1 (sometimes denoted as d+) used in the Black-Scholes formula
    """

    d_1 = (np.log(S_t/K) + (r + 0.5 * (sigma**2)) * time_till_mat) / (sigma * np.sqrt(time_till_mat))

    return d_1


def d2(S_t, K, sigma, r, time_till_mat):
    """
    :param S_t:           Current spot price
    :param K:             Strike price
    :param sigma:         Volatility
    :param r:             Current short rate
    :param time_till_mat: Length of time remaining in the option

    :return: Returns the parameter d2 (sometimes denoted as d-) used in the Black-Scholes formula
    """

    d_2 = d1(S_t, K, sigma, r, time_till_mat) - sigma * np.sqrt(time_till_mat)

    return d_2


def delta(option_type, S_t, K, sigma, r, time_till_mat):
    """
    :param option_type:   C for Call or P for Put
    :param S_t:           Current spot price
    :param K:             Strike price
    :param sigma:         Volatility
    :param r:             Current short rate
    :param time_till_mat: Length of time remaining in the option

    :return: Returns the corresponding delta for an option whose parameters match the input
             parameters to this function.
    """

    if option_type == 'C':
        return norm.cdf(d1(S_t, K, sigma, r, time_till_mat))
    elif option_type == 'P':
        return norm.cdf(d1(S_t, K, sigma, r, time_till_mat)) - 1
    else:
        print("Invalid type")


def gamma(option_type, S_t, K, sigma, r, time_till_mat):
    """
    :param option_type:   C for Call or P for Put
    :param S_t:           Current spot price
    :param K:             Strike price
    :param sigma:         Volatility
    :param r:             Current short rate
    :param time_till_mat: Length of time remaining in the option

    :return: Returns the corresponding gamma for an option whose parameters match the input
             parameters to this function.
    """

    if option_type == 'C' or option_type == 'P':
        return norm.pdf(d1(S_t, K, sigma, r, time_till_mat)) * (1 / (sigma * S_t * np.sqrt(time_till_mat)))
    else:
        print("Invalid type")


def vega(option_type, S_t, K, sigma, r, time_till_mat):
    """
    :param option_type:   C for Call or P for Put
    :param S_t:           Current spot price
    :param K:             Strike price
    :param sigma:         Volatility
    :param r:             Current short rate
    :param time_till_mat: Length of time remaining in the option

    :return: Returns the corresponding vega for an option whose parameters match the input
             parameters to this function.
    """

    if option_type == 'C' or option_type == 'P':
        return norm.pdf(d1(S_t, K, sigma, r, time_till_mat)) * np.sqrt(time_till_mat) * S_t
    else:
        print("Invalid type")


def rho(option_type, S_t, K, sigma, r, time_till_mat):
    """
    :param option_type:   C for Call or P for Put
    :param S_t:           Current spot price
    :param K:             Strike price
    :param sigma:         Volatility
    :param r:             Current short rate
    :param time_till_mat: Length of time remaining in the option

    :return: Returns the corresponding rho for an option whose parameters match the input
             parameters to this function.
    """

    if option_type == 'C':
        return norm.cdf(d2(S_t, K, sigma, r, time_till_mat)) * time_till_mat * np.exp(-r * time_till_mat) * K
    elif option_type == 'P':
        return (norm.cdf(d2(S_t, K, sigma, r, time_till_mat)) - 1) * time_till_mat * np.exp(-r * time_till_mat) * K


def theta(option_type, S_t, K, sigma, r, time_till_mat):
    """
    :param option_type:   C for Call or P for Put
    :param S_t:           Current spot price
    :param K:             Strike price
    :param sigma:         Volatility
    :param r:             Current short rate
    :param time_till_mat: Length of time remaining in the option

    :return: Returns the corresponding theta for an option whose parameters match the input
             parameters to this function.
    """

    if option_type == 'C':
        first_term = (-1) * r * np.exp((-1) * r * time_till_mat) * K * norm.cdf(d2(S_t, K, sigma, r, time_till_mat))
        second_term = (((-1) * sigma * S_t) / (2 * np.sqrt(time_till_mat))) * norm.pdf(d1(S_t, K, sigma, r, time_till_mat))
        sum_of_terms = first_term + second_term

        return sum_of_terms
    elif option_type == 'P':
        first_term = r * np.exp((-1) * r * time_till_mat) * K * (1 - norm.cdf(d2(S_t, K, sigma, r, time_till_mat)))
        second_term = (((-1) * sigma * S_t) / (2 * np.sqrt(time_till_mat))) * norm.pdf(d1(S_t, K, sigma, r, time_till_mat))

        sum_of_terms = first_term + second_term
        return sum_of_terms


# --------------------------------------- Functions for Plotting --------------------------------------- #

def plot_greek(greek_func, option_type, sigma):
    """

    :param greek_func:  One of the greek functions defined above, e.g. delta, gamma, vega, rho, or theta
    :param option_type: C for Call or P for Put
    :param sigma:       Volatility, included since our HW asks us to consider 2 values for sigma

    :return: Does not return anything, purpose of function is to create plots
    """

    initial_spot = np.linspace(40, 160, 240)

    for K in [80, 90, 100, 110, 120]:
        greek_vec = np.vectorize(greek_func)(option_type, initial_spot, K, sigma, 0.02, 30 / 365)
        plt.plot(initial_spot, greek_vec, label="K = {}".format(K))

    if option_type == 'C':
        plt.title("Call Option {} (sigma = {})".format(greek_func.__name__, sigma))
    elif option_type == 'P':
        plt.title("Put Option {} (sigma = {})".format(greek_func.__name__, sigma))

    plt.xlabel("Initial Stock Price S_t")
    plt.ylabel("{}".format(greek_func.__name__))
    plt.legend(loc='upper left')
    plt.show()


def plot_greeks(sigma):
    """

    :param sigma: Volatility, included since our HW asks us to consider 2 values for sigma

    :return: Does not return anything, purpose of function is to create plots
    """

    # For each of the greek functions defined above, e.g. delta, gamma, vega, rho, and theta, we
    # apply the plot_greek function -- implemented immediately above this function -- which
    # will plot each of the greeks as functions of the current spot price S_t and for 5 different
    # strike prices.
    for greek_func in [delta, gamma]:#, vega, rho, theta]:
        plot_greek(greek_func, 'C', sigma)
        plot_greek(greek_func, 'P', sigma)


# ---------------- Question 1(b) ---------------- #
plot_greeks(0.15)
plot_greeks(0.25)


# ---------------- Question 1(c) ---------------- #

def atm_straddle_greeks(K):
    straddle_greeks_vec = []
    for greek in [delta, gamma, vega, rho, theta]:
        call_greek = greek('C', K, K, 0.15, 0.02, 30/365)
        put_greek = greek('P', K, K, 0.15, 0.02, 30/365)

        straddle_greeks_vec.append(call_greek + put_greek)

    return straddle_greeks_vec


def print_straddle_greeks(K):
    print("Straddle delta: {}".format(atm_straddle_greeks(K)[0]))
    print("Straddle gamma: {}".format(atm_straddle_greeks(K)[1]))
    print("Straddle vega: {}".format(atm_straddle_greeks(K)[2]))
    print("Straddle rho: {}".format(atm_straddle_greeks(K)[3]))
    print("Straddle theta: {}".format(atm_straddle_greeks(K)[4]))


print("------------ Greeks for ATM Straddle at Various Strikes ------------")
for strike in [80, 90, 100, 110, 120]:
    print("Strike = ", strike)
    print_straddle_greeks(strike)
    print("\n")


# ------------------ Question 3 ------------------ #
oct_price_of_call1 = 3.1 * norm.cdf(d1(3.1, 4.3, 0.27, 0.0217, 30/365)) - \
                     4.3 * np.exp(-0.0217*(30/365)) * norm.cdf(d2(3.1, 4.3, 0.27, 0.0217, 30/365))
nov_price_of_call1 = 4.5 * norm.cdf(d1(4.5, 4.3, 1.05, 0.0225, 30/365)) - \
                     4.3 * np.exp(-0.0217*(30/365)) * norm.cdf(d2(4.5, 4.3, 1.05, 0.0225, 30/365))
print("October price (K = 4): {:.7f}".format(oct_price_of_call1))
print("November price (K = 4): {:.7f}".format(nov_price_of_call1))
