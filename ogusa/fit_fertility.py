import os
import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.ndimage.interpolation import shift
import math
from math import e
import matplotlib.pyplot as plt

cur_path = '/Users/adamalexanderoppenheimer/Desktop/OG-JPN/ogusa/'
os.chdir(cur_path)
datadir = 'data/demographic/'
fert_dir = datadir + 'jpn_fertility.csv'
mort_dir = datadir + 'jpn_mortality.csv'
pop_dir = datadir + 'jpn_population.csv'

def select_fert_data(fert, set_zeroes=False):
    new_fert = fert[fert['AgeDef'] == 'ARDY']
    new_fert = new_fert[new_fert['Collection'] == 'HFD']
    new_fert = new_fert[(new_fert['RefCode'] == 'JPN_11')]
    new_fert.drop(['AgeDef', 'Collection', 'RefCode'], axis=1, inplace=True)
    new_fert.columns = ['Year', 'Age', 'Values']
    if set_zeroes:
        new_fert['Values'][new_fert['Age'] == 14] = 0
        new_fert['Values'][new_fert['Age'] == 15] = 0
        new_fert['Values'][new_fert['Age'] == 49] = 0
        new_fert['Values'][new_fert['Age'] == 50] = 0
    return new_fert.astype(float)

fert_data = pd.read_csv(fert_dir, sep=r',\s*',\
    usecols=['Year1', 'Age', 'ASFR', 'AgeDef',\
                    'Collection', 'RefCode'])
fert_data = select_fert_data(fert_data)

fert_data['Cohort'] = fert_data['Year'] - fert_data['Age']
fert_data.drop('Year', axis=1, inplace=True)
fert_data = fert_data.pivot(index='Age', columns='Cohort', values='Values')

def array_add(array_list):
    if len(array_list) == 1:
        return array_list[0]
    array_sum = array_list[0]
    for i in range(1, len(array_list)):
        array_sum += array_list[i]
    return array_sum

def rolling_avg_age(data, roll):
    if roll >= len(data) // 2:
        print('Stop trying to roll half or more of the values!')
        return False

    data_orig = np.copy(data)

    forwards = []
    backwards = []

    for i in range(1, roll + 1):
        forwards.append(shift(data, i, cval=np.NaN))
        backwards.append(shift(data, -i, cval=np.NaN))

    data = (array_add(forwards) + array_add(backwards) + data_orig) / (len(forwards) + len(backwards) + 1)

    for i in range(roll, -1, -1):
        if i > 0:
            relevant_forwards = (array_add(forwards[:i]) + array_add(backwards) + data_orig) / (i + len(backwards) + 1)
            relevant_backwards = (array_add(backwards[:i]) + array_add(forwards) + data_orig) / (i + len(forwards) + 1)

        else:
            relevant_forwards = (array_add(backwards) + data_orig) / (i + len(backwards) + 1)
            relevant_backwards = (array_add(forwards) + data_orig) / (i + len(forwards) + 1)

        nan_vals = np.isnan(data)
        data[nan_vals] = relevant_forwards[nan_vals]

        nan_vals = np.isnan(data)
        data[nan_vals] = relevant_backwards[nan_vals]
    return data

def rolling_avg_year(year, roll):
    years = []
    for yr in range(year - roll, year + roll + 1):
        try:
            years.append(fert_data[yr])
        except:
            pass
    tot = len(years)
    avg = array_add(years) / tot
    return avg.dropna()

alphas = []
betas = []
ms = []
scales = []
for year in range(1975,2001):
    #Fit data for 1980 cohort
    #fert_1980 = fert_data[year].dropna()
    
    #Take 5 years rolling average
    fert_1980 = rolling_avg_year(year, 0)
    scales.append(np.sum(fert_1980))

    fert_1980 = fert_1980 / fert_1980.sum()
    count = fert_1980.shape[0]
    mean = 30#fert_1980.mean()
    median = 30#np.median(fert_1980)
    std = 5#fert_1980.std()
    var = 25#fert_1980.var()

    def gamma_fun_pdf(xvals, alpha, beta):
        pdf_vals = ( xvals ** (alpha - 1) * e ** ( - xvals / beta ) )/\
            ( beta ** alpha * math.gamma(alpha) )
        return pdf_vals

    def crit_b(params, *args):
        alpha, beta = params
        xvals, dist_pts = args
        guess = gamma_fun_pdf(dist_pts, alpha, beta)
        xvals = xvals * guess.sum() / xvals.sum() #Restandardize data
        diff = np.sum((xvals - guess) ** 2)
        return diff

    if year == 1975:
        beta_0 = var/mean
        alpha_0 = mean/beta_0
    params_init = np.array([alpha_0, beta_0])
    dist_pts = np.linspace(14, 34, count)

    results_cstr = opt.minimize(crit_b, params_init,\
                    args=(np.array(fert_1980), dist_pts), method="L-BFGS-B",\
                    bounds=((1e-10, None), (1e-10, None)), options={'eps':1})
    alpha_MLE_b, beta_MLE_b = results_cstr.x

    print("alpha_MLE_b=", alpha_MLE_b, " beta_MLE_b=", beta_MLE_b)

    #plt.plot(dist_pts, gamma_fun_pdf(dist_pts, alpha_MLE_b, beta_MLE_b) / gamma_fun_pdf(dist_pts, alpha_MLE_b, beta_MLE_b).sum(),\
    #        linewidth=2, label="Gamma", color="r")
    #plt.plot(fert_1980 / fert_1980.sum())
    #plt.show()

    ####################################################################################
    def gen_gamma_fun_pdf(xvals, alpha, beta, m):
            pdf_vals = (m * e ** ( - ( xvals / beta ) ** m ))/\
                    (xvals * math.gamma(alpha / m)) *\
                    (xvals / beta) ** alpha
            return pdf_vals

    def log_sum_c(xvals, alpha, beta, m):
            log_vals = np.log(m) + (alpha - 1) * np.log(xvals) -\
                    (xvals / beta) ** m - alpha * np.log(beta) -\
                    np.log(math.gamma(alpha / m))
            return log_vals.sum()

    def crit_c(params, *args):
            alpha, beta, m = params
            xvals, dist_pts = args
            guess = gen_gamma_fun_pdf(dist_pts, alpha, beta, m)
            xvals = xvals * guess.sum() / xvals.sum() #Restandardize data
            diff = np.sum((xvals - guess) ** 2)
            return diff

    fert_1980 = fert_1980 / fert_1980.sum()

    alpha_0 = alpha_MLE_b
    beta_0 = beta_MLE_b
    if year == 1975:
        m_0 = 1
    params_init = np.array([alpha_0, beta_0, m_0])

    results_cstr = opt.minimize(crit_c, params_init,\
                                args=(fert_1980, dist_pts), method="L-BFGS-B",\
                                bounds=((1e-10, None), (1e-10, None),\
                                (1e-10, None)), tol=1e-100, options={'eps':1e-10})
    alpha_MLE_c, beta_MLE_c, m_MLE_c = results_cstr.x

    #Use previous year's results as start values for next year
    alpha_0 = alpha_MLE_c
    beta_0 = beta_MLE_c
    m_0 = m_MLE_c

    print("alpha_MLE_c=", alpha_MLE_c, " beta_MLE_c=", beta_MLE_c, " m_MLE_c=", m_MLE_c)

    #plt.plot(dist_pts, gen_gamma_fun_pdf(dist_pts, alpha_MLE_c, beta_MLE_c, m_MLE_c) / gen_gamma_fun_pdf(dist_pts, alpha_MLE_c, beta_MLE_c, m_MLE_c).sum(),\
    #        linewidth=2, label="Generalized Gamma", color="r")
    #plt.plot(fert_1980 / fert_1980.sum())
    #plt.show()

    alphas.append(alpha_MLE_c)
    betas.append(beta_MLE_c)
    ms.append(m_MLE_c)

alphas = np.array(alphas)
betas = np.array(betas)
ms = np.array(ms)
scales = np.array(scales)

years = np.linspace(1975, 2000, 26)
lines = []
lines.append(plt.plot(years, alphas))
lines.append(plt.plot(years, betas))
lines.append(plt.plot(years, ms))
lines.append(plt.plot(years, scales))

lines[0][0].set_label('Alpha')
lines[1][0].set_label('Beta')
lines[2][0].set_label('M')
lines[3][0].set_label('Scale')

plt.legend()

plt.show()
