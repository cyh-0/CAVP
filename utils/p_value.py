import numpy as np
from scipy.stats import norm
from scipy.stats import t

auc_a = [84.80, 82.51, 82.77]  # AUC of Model A from three runs
auc_b = [84.04, 84.53, 82.54]  # AUC of Model B from three runs
n = 3  # number of runs


def z_test(auc_a, auc_b, n):
    print("RUNNING Z-TEST")
    mean_a = np.mean(auc_a)
    mean_b = np.mean(auc_b)
    std_a = np.std(auc_a, ddof=1)  # ddof=1 to use Bessel's correction
    std_b = np.std(auc_b, ddof=1)

    SE = np.sqrt(((std_a**2) / n) + ((std_b**2) / n))
    z = (mean_a - mean_b) / SE

    p_value = norm.sf(z)  # one-tailed p-value
    alpha = 0.05  # significance level

    print(p_value)

    if p_value < alpha:
        print("The AUC of Model A is significantly greater than the AUC of Model B.")
    else:
        print(
            "There is insufficient evidence to conclude that the AUC of Model A is significantly greater than the AUC of Model B."
        )


def t_test(auc_a, auc_b, n):
    print("RUNNING T-TEST")
    mean_a = np.mean(auc_a)
    mean_b = np.mean(auc_b)
    std_a = np.std(auc_a, ddof=1)  # ddof=1 to use Bessel's correction
    std_b = np.std(auc_b, ddof=1)
    s_pooled = np.sqrt(((n - 1) * (std_a**2) + (n - 1) * (std_b**2)) / (2 * n - 2))
    t_statistic = (mean_a - mean_b) / (s_pooled * np.sqrt(2 / n))
    p_value = t.sf(t_statistic, 2 * n - 2)  # one-tailed p-value
    alpha = 0.05  # significance level
    print(p_value)

    if p_value < alpha:
        print("The AUC of Model A is significantly greater than the AUC of Model B.")
    else:
        print(
            "There is insufficient evidence to conclude that the AUC of Model A is significantly greater than the AUC of Model B."
        )


z_test(auc_a, auc_b, n)
t_test(auc_a, auc_b, n)
