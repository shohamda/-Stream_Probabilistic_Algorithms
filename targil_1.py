import numpy as np
import math
import mmh3
import random
import matplotlib.pyplot as plt


# morris alpha version
def Morris_Alpha(stream):
    x = 0
    for element in enumerate(stream):
        d = 1 / (np.power(2, x))
        r = np.random.random()
        if r < d:
            x += 1
    return (np.power(2, x) - 1)


# morris beta version
def Morris_Beta(stream, counters):
    s = []
    for i in range(counters):
        s.append(Morris_Alpha(stream))
    return np.mean(s)


# flajolet_martin_f0
def fm_f0(stream):
    x = 1
    seed = random.randint(0, 10000)
    for element in stream:
        h = mmh3.hash(element, seed=seed, signed=False) / (2 ** 32 - 1)
        if h < x:
            x = h
    return x


# flajolet_martin_beta
def fm_beta(stream, counters):
    s = []
    for i in range(counters):
        s.append(fm_f0(stream))
    return ((1 / (np.mean(s))) - 1)


# flajolet_martin_final
def fm_final(stream, final_counter, beta_counters):
    z = []
    for i in range(0, final_counter):
        z.append(fm_beta(stream, beta_counters))
    return np.median(z)


if __name__ == "__main__":
    random.seed(149)  # set a random seed to be able reproduse results
    uniques = 10000
    stream = np.random.randint(uniques, size=10000)  # A-2: synthetic dataset, 10,000 unique and 1,000,000 elements

    # answer 3
    copies = [30, 50, 100, 125, 150]  # number of copies for statistical significance * note: at least 100 copies
    morris_est = []
    fm0_est = []
    morris_normal_var = []
    fm0_normal_var = []
    for a in copies:
        morris_estimator = [0]
        fm0_estimator = [0]
        for i in range(a):
            random.seed(i)  # change the seed for 100 iterations
            morris_estimator.append(Morris_Alpha(stream))
            fm0_estimator.append((1 / fm_f0(stream)))
        morris_est.append(np.mean(morris_estimator))  # normalized estimator
        fm0_est.append(np.mean(fm0_estimator))  # normalized estimator
        morris_normal_var.append((np.var(morris_estimator) / len(stream)))  # normalized var
        fm0_normal_var.append((np.var(fm0_estimator) / len(stream)))  # normalized var
        print('morris estimator for', a, 'copies is:', morris_est[-1])
        print('morris var for', a, 'copies is: ', morris_normal_var[-1])
        print('FM estimator for', a, 'copies is:', fm0_est[-1])
        print('fm var for', a, 'copies is:', fm0_normal_var[-1])

    # morris alpha and fm 0 estimators and var graphs
    plt.hist(morris_est)
    plt.xlabel('estimation')
    plt.ylabel('copies')
    plt.title('morris estimations per copies')
    plt.show()

    plt.hist(fm0_est)
    plt.xlabel('estimation')
    plt.ylabel('copies')
    plt.title('fm0 estimations per copies')
    plt.show()

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('copies')
    ax1.set_ylabel('morris alpha estimators', color=color)
    ax1.plot(list(copies), morris_est, color=color)
    plt.axhline(y=len(stream), color=color, linestyle='--')  # accuracy
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('fm0 estimators', color=color)  # we already handled the x-label with ax1
    ax2.plot(list(copies), fm0_est, color=color)
    plt.axhline(y=uniques, color='blue', linestyle='--')  # accuracy
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('morris alpha and fm0 estimators per copies')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    # answer 4.i.a+b (morris beta and fm beta)
    new_copies = 100
    beta_counters = np.array([10, 50, 100])
    final_counters = np.array([10, 50])

    morris_beta_est = []
    fm_beta_est = []
    mem_size_morris = []
    mem_size_fm_beta = []
    fm_beta_var = []
    fm_beta_error = []
    morris_beta_var = []
    morris_beta_error = []
    delta = 0.01

    for i in beta_counters:  # i = 10,50,100
        morris_beta_estimator = []
        fm_beta_estimator = []
        for a in range(new_copies):
            morris_temp_estimator = (Morris_Beta(stream, i))
            morris_beta_estimator.append(morris_temp_estimator)
            fm_beta_temp_estimator = (fm_beta(stream, i))
            fm_beta_estimator.append(fm_beta_temp_estimator)
        morris_beta_est.append(np.mean(morris_beta_estimator))
        fm_beta_est.append(np.mean(fm_beta_estimator))
        morris_beta_var.append(np.var(morris_beta_estimator) / len(stream))
        fm_beta_var.append(np.var(fm_beta_estimator) / len(stream))
        morris_beta_error.append(np.sqrt(1 / (delta * i)))
        fm_beta_error.append((np.sqrt(1 / (delta * i))))
    print('morris beta estimators: counters:', beta_counters, 'estimators:', morris_beta_est)
    print('fm beta estimators: counters:', beta_counters, 'estimators:', fm_beta_est)
    print('morris beta vars: counters:', beta_counters, 'vars:', morris_beta_var)
    print('fm beta vars: counters:', beta_counters, 'vars:', fm_beta_est)
    print('morris beta errors: counters:', beta_counters, 'errors:', morris_beta_error)
    print('fm beta errors: counters:', beta_counters, 'errors:', fm_beta_error)
    # calculate morris beta and fm beta memory storage
    mem_size_morris_beta = [x * math.log(math.log(len(stream))) for x in beta_counters]
    print('memory of morris beta:', mem_size_morris_beta)
    mem_size_fm_beta = [1 / (delta * (error ** 2)) for error in fm_beta_error]
    print('memory of fm beta:', mem_size_fm_beta)

    # morris beta and fm beta graphs
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('copies')
    ax1.set_ylabel('morris beta', color=color)
    ax1.plot(list(beta_counters), morris_beta_est, color=color)
    plt.axhline(y=len(stream), color=color, linestyle='--')  # accuracy
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('fm beta', color=color)  # we already handled the x-label with ax1
    ax2.plot(list(beta_counters), fm_beta_est, color=color)
    plt.axhline(y=10, color='blue', linestyle='--')  # accuracy
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('morris beta and fm beta estimators per copies')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    # normalized var graph
    var_fig, var_ax1 = plt.subplots()

    color = 'tab:red'
    var_ax1.set_xlabel('copies')
    var_ax1.set_ylabel('morris var', color=color)
    var_ax1.plot(list(beta_counters), morris_beta_var, color=color)
    var_ax1.tick_params(axis='y', labelcolor=color)

    var_ax2 = var_ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    var_ax2.set_ylabel('fm beta var', color=color)  # we already handled the x-label with ax1
    var_ax2.plot(list(beta_counters), fm_beta_var, color=color)
    var_ax2.tick_params(axis='y', labelcolor=color)

    plt.title('morris beta and fm beta var per copies')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    # memory size graph
    mem_fig, mem_ax1 = plt.subplots()

    color = 'tab:red'
    mem_ax1.set_xlabel('copies')
    mem_ax1.set_ylabel('morris memory', color=color)
    mem_ax1.plot(list(beta_counters), mem_size_morris_beta, color=color)
    mem_ax1.tick_params(axis='y', labelcolor=color)

    mem_ax2 = mem_ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    mem_ax2.set_ylabel('fm beta memory', color=color)  # we already handled the x-label with ax1
    mem_ax2.plot(list(beta_counters), mem_size_fm_beta, color=color)
    mem_ax2.tick_params(axis='y', labelcolor=color)

    plt.title('morris beta and fm beta memory per copies')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    # answer 4.i.c
    copies = 100
    final_res = []
    fm_final_res = []
    fm_final_var = []
    fm_final_error = []

    for i in final_counters:
        for t in beta_counters:
            z = []
            for k in range(copies):
                fm_final_temp_estimator = fm_final(stream, i, t)
                z.append(fm_final_temp_estimator)

            fm_final_res.append(np.mean(z))
            fm_final_var.append(np.var(z) / len(stream))
            fm_final_error.append(math.log(1 / delta) / (i * t))

    print('fm final estimators: counters:', final_counters, 'estimators:', fm_final_res)
    print('fm final vars: counters:', final_counters, 'var:', fm_final_var)
    print('fm final error: counters:', final_counters, 'error:', fm_final_error)

    # calculate memory size
    mem_size_fm_final = [math.log(1 / delta) / (error ** 2) for error in fm_final_error]
    print('memory of fm final:', mem_size_fm_final)

    # print fm final graphs
    x = [f'({a},{b})' for a in final_counters for b in beta_counters]  # find that online
    print(x)
    plt.plot(x, fm_final_res)
    plt.title('estimation per copies')
    plt.xlabel('final copies')
    plt.ylabel('estimation')
    plt.axhline(y=uniques, linestyle='--', color='red')
    plt.show()

    plt.plot(x, fm_final_var)
    plt.title('varper copies')
    plt.xlabel('final copies')
    plt.ylabel('normalized var')
    plt.show()

    plt.plot(x, fm_final_error)
    plt.title('error per copies')
    plt.xlabel('final copies')
    plt.ylabel('eroor')
    plt.show()

    plt.plot(x, mem_size_fm_final)
    plt.title('memory size per copies')
    plt.xlabel('final copies')
    plt.ylabel('memory size')
    plt.show()

    # relative error (4.ii)
    delta = 0.01
    # morris beta s = 1 / (delta * epsilon ** 2)
    morris_epsilon = [0] * len(beta_counters)
    morris_epsilon = pow(1 / (beta_counters * delta), -2)

    # fm beta s = 1 / (delta * epsilon ** 2)
    fm_beta_epsilon = [0] * len(beta_counters)
    fm_beta_epsilon = pow(1 / (beta_counters * delta), -2)

    # fm final s = 1 / (epsilon ** 2) log (1 / delta)
    fm_final_epsilon = [0] * len(final_counters)
    fm_final_epsilon = (1 / final_counters) * math.log(1 / delta)

    print('delta = ', delta, '\nmorris epsilon ([copies],[epsilon]):', beta_counters, morris_epsilon,
          '\nfm beta epsilon ([copies],[epsilon]):', beta_counters, fm_beta_epsilon,
          '\nfm final epsilon ([copies],[epsilon]):'
          , final_counters, fm_final_epsilon)
