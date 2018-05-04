import numpy as np
import scipy.stats


def lhood_dist(x, param, dist_name, threshold=1e-10):
    if dist_name == "weibull_min":
        lh = scipy.stats.weibull_min.pdf(x, c=param[0], loc=param[1], scale=param[2])
    elif dist_name == "exponnorm":
        lh = scipy.stats.exponnorm.pdf(x, K=param[0], loc=param[1], scale=param[2])
    elif dist_name == "expon":
        lh = scipy.stats.expon.pdf(x, loc=param[0], scale=param[1])
    elif dist_name == "lognorm":
        lh = scipy.stats.lognorm.pdf(x, s=param[0], loc=param[1], scale=param[2])
    elif dist_name == "norm":
        lh = scipy.stats.norm.pdf(x, loc=param[0], scale=param[1])
    elif dist_name == "invgauss":
        lh = scipy.stats.invgauss.pdf(x, mu=param[0], loc=param[1], scale=param[2])
    elif dist_name == "weibull_min_floc":
        lh = scipy.stats.weibull_min.pdf(x, c=param[0], loc=0, scale=param[1])
    elif dist_name == "expon_floc":
        lh = scipy.stats.expon.pdf(x, loc=0, scale=param[0])
    elif dist_name == "lognorm_floc":
        lh = scipy.stats.lognorm.pdf(x, s=param[0], loc=0, scale=param[1])
    elif dist_name == "invgauss_floc":
        lh = scipy.stats.invgauss.pdf(x, mu=param[0], loc=0, scale=param[1])
    elif dist_name == "beta_floc":
        lh = scipy.stats.beta.pdf(x, a=param[0], b=param[1], loc=0, scale=param[2])
    else:
        raise ValueError(dist_name + " does not exist")

    lh = np.array(lh)
    lh = np.nan_to_num(lh)
    lh[lh < threshold] = threshold

    lh *= max(x)

    return lh


def nllh(x, param, dist_name, threshold=1e-10):
    llh = np.log(lhood_dist(x=x, param=param, dist_name=dist_name, threshold=threshold))
    if not np.isfinite(llh).all():
        print("nllh not finite", dist_name, param, x)
        return 100
    nllh = -np.sum(llh) / x.shape[0]
    return nllh


def fit_dist(x, dist_name):
    if dist_name == "weibull_min":
        param = scipy.stats.weibull_min.fit(x)
    elif dist_name == "exponnorm":
        param = scipy.stats.exponnorm.fit(x)
        if param[0] < 0.03:
            print("Thresholding K")
            param = (0.03, param[1], param[2])
    elif dist_name == "expon":
        param = scipy.stats.expon.fit(x)
    elif dist_name == "lognorm":
        param = scipy.stats.lognorm.fit(x)
    elif dist_name == "norm":
        param = scipy.stats.norm.fit(x)
    elif dist_name == "invgauss":
        param = scipy.stats.invgauss.fit(x)
    elif dist_name == "weibull_min_floc":
        p = scipy.stats.weibull_min.fit(x, floc=0)
        param = [p[0], p[2]]
    elif dist_name == "expon_floc":
        p = scipy.stats.expon.fit(x, floc=0)
        param = [p[1], ]
    elif dist_name == "lognorm_floc":
        p = scipy.stats.lognorm.fit(x, floc=0)
        param = [p[0], p[2]]
    elif dist_name == "invgauss_floc":
        p = scipy.stats.invgauss.fit(x, floc=0)
        param = [p[0], p[2]]
    elif dist_name == "beta_floc":
        p = scipy.stats.beta.fit(x, floc=0)
        param = [p[0], p[1], p[3]]
    else:
        raise ValueError(dist_name + " does not exist")

    return np.array(param)


def cdf_dist(x, param, dist_name):
    if dist_name == "weibull_min":
        cdf = scipy.stats.weibull_min.cdf(x, c=param[0], loc=param[1], scale=param[2])
    elif dist_name == "exponnorm":
        cdf = scipy.stats.exponnorm.cdf(x, K=param[0], loc=param[1], scale=param[2])
    elif dist_name == "expon":
        cdf = scipy.stats.expon.cdf(x, loc=param[0], scale=param[1])
    elif dist_name == "lognorm":
        cdf = scipy.stats.lognorm.cdf(x, s=param[0], loc=param[1], scale=param[2])
    elif dist_name == "norm":
        cdf = scipy.stats.norm.cdf(x, loc=param[0], scale=param[1])
    elif dist_name == "invgauss":
        cdf = scipy.stats.invgauss.cdf(x, mu=param[0], loc=param[1], scale=param[2])
    elif dist_name == "weibull_min_floc":
        cdf = scipy.stats.weibull_min.cdf(x, c=param[0], loc=0, scale=param[1])
    elif dist_name == "expon_floc":
        cdf = scipy.stats.expon.cdf(x, loc=0, scale=param[0])
    elif dist_name == "lognorm_floc":
        cdf = scipy.stats.lognorm.cdf(x, s=param[0], loc=0, scale=param[1])
    elif dist_name == "invgauss_floc":
        cdf = scipy.stats.invgauss.cdf(x, mu=param[0], loc=0, scale=param[1])
    elif dist_name == "beta_floc":
        cdf = scipy.stats.beta.cdf(x, a=param[0], b=param[1], loc=0, scale=param[2])
    else:
        raise ValueError(dist_name + " does not exist")

    return cdf


def kstest(x, param, dist_name):
    if dist_name in ["weibull_min", "exponnorm", "expon", "lognorm", "norm", "invgauss"]:
        d, p = scipy.stats.kstest(x, dist_name, param)
    elif dist_name == "weibull_min_floc":
        dist = scipy.stats.weibull_min(c=param[0], loc=0, scale=param[1])
        d, p = scipy.stats.kstest(x, dist.pdf)
    elif dist_name == "expon_floc":
        d, p = scipy.stats.kstest(x, "expon", [0, param[0]])
    elif dist_name == "lognorm_floc":
        d, p = scipy.stats.kstest(x, "lognorm", [param[0], 0, param[1]])
    elif dist_name == "invgauss_floc":
        d, p = scipy.stats.kstest(x, "invgauss", [param[0], 0, param[1]])
    elif dist_name == "beta_floc":
        d, p = scipy.stats.kstest(x, "beta", [param[0], param[1], 0, param[2]])
    else:
        raise ValueError(dist_name + " does not exist")

    return p



