import numpy as np
import math
import mmh3
import random
import statistics
import matplotlib.pyplot as plt

#morris alpha version
def Morris_Alpha(stream):
    x = 0
    for element in stream:
        d = 1 / (math.pow(2,x))
        r = random.uniform(0, 1)
        if r < d:
            x += 1
    return (math.pow(2,x) -1)

#morris beta version
def Morris_Beta(stream, counters):
  s = [0] * counters
  for i in range(len(s)):
    s[i]=Morris_Alpha(stream)
  return np.mean(s)

#flajolet_martin_f0
def fm_f0(stream):
    x = 1
    seed = random.randint(0,10000)
    for element in stream:
        h = mmh3.hash(element,seed=seed, signed=False) / (2**32 -1)
        if h < x:
            x = h
    return x

# flajolet_martin_beta
def fm_beta(stream, counters):
    s = [0] * counters
    for i in range(len(s)):
        s[i] = fm_f0(stream)
    return ((1 / (np.mean(s))) - 1)

#flajolet_martin_final
def fm_final(stream, beta_counters):
  z = [0] * beta_counters
  for i in range(len(z)):
    z[i] = fm_beta(stream,beta_counters)
  return statistics.median(z)

if __name__ == "__main__":
    random.seed(1000)  # set a random seed to be able reproduse results
    stream = np.random.randint(10000, size=10000)  # A-2: synthetic dataset, 10,000 unique and 1,000,000 elements

    # answer 3
    COPIES = [10, 25, 50, 100] # number of copies for statistical significance
    for a in COPIES:
        morris_estimator = [0] * a
        FM0_estimator = [0] * a
        for i in range(a):
            random.seed(i)  # change the seed for 100 iterations
            morris_estimator[i] = Morris_Alpha(stream)
            FM0_estimator[i] = (1 / fm_f0(stream))
        morris_est = np.mean(morris_estimator)  # normalized estimator
        FM0_est = np.mean(FM0_estimator)  # normalized estimator
        print('morris_estimator for', a, 'copies', morris_est)
        print('FM_estimator for',a ,'copies', FM0_est)
        morris_normal_var = np.var(morris_estimator / len(stream))  # normalized var
        FM0_normal_var = np.var(FM0_estimator / len(stream))  # normalized var
        print('var_morris for',a ,'copies', morris_normal_var)
        print('var_FM for',a ,'copies', FM0_normal_var)

    #answer 4.i
    beta_counters = np.array([4, 5, 6])
    final_counters = np.array([2, 4])

    morris_estimator = [0] * len(beta_counters)
    FM_beta_estimator = [0] * len(beta_counters)
    mem_size_morris = [0] * len(beta_counters)
    mem_size_fm_beta = [0] * len(beta_counters)
    final_res = []

    # answer 4.i.a+b
    fm_beta_res = []
    fm_beta_var = []
    fm_beta_error = []
    morris_beta_res = []
    morris_beta_var = []
    morris_beta_error =[]


    for k in range(len(beta_counters)): # k = 0,1,2
        # calculate morris beta and fm beta
        for i in beta_counters: # i = 10,50,100
            morris_estimator[k] = Morris_Beta(stream, i)
            FM_beta_estimator[k] = fm_beta(stream, i)

        fm_beta_res.append(np.mean(FM_beta_estimator))
        fm_beta_var.append(np.var(FM_beta_estimator) / len(stream))
        fm_beta_error.append((np.sqrt(1 / (DELTA * i))))

        morris_beta_res.append(np.mean(morris_estimator))
        morris_beta_var.append(np.var(morris_estimator) / len(stream))
        morris_beta_error.append(np.sqrt(1 / (DELTA * i)))

    # answer 4.i.c
    fm_final_res = []
    fm_final_var = []
    fm_final_error = []

    for i in final_counters:
        z = [0] * i
        for t in beta_counters:
            for k in range(i):
                z[k] = fm_final(stream, t)
            final_res.append(np.average(z))

    fm_final_res.append(np.mean(final_res))
    fm_final_var.append(np.var(final_res) / len(stream))
    fm_final_error.append(math.log(1 / DELTA) / (i * t))

    #relative error (4.ii)
    delta = 0.01
    #morris beta s = 1 / (delta * epsilon ** 2)
    morris_epsilon = [0] * len(beta_counters)
    morris_epsilon = pow(1/(beta_counters * delta),-2)

    # fm beta s = 1 / (delta * epsilon ** 2)
    fm_beta_epsilon = [0] * len(beta_counters)
    fm_beta_epsilon = pow(1/(beta_counters * delta),-2)

    # fm final s = 1 / (epsilon ** 2) log (1 / delta)
    fm_final_epsilon = [0] * len(final_counters)
    fm_final_epsilon = (1 /final_counters) * math.log(1/delta)

    print('delta = ', delta, '\nmorris epsilon ([copies],[epsilon]):', beta_counters, morris_epsilon,
          '\nfm beta epsilon ([copies],[epsilon]):',beta_counters ,fm_beta_epsilon,'\nfm final epsilon ([copies],[epsilon]):'
          ,final_counters,fm_final_epsilon)


    # calculate memory size for each copy
    mem_size_morris = beta_counters * math.log(math.log(len(stream)))
    mem_size_fm_beta = beta_counters * math.log(math.log(len(stream))) + 5 #!!!! need to find out the real memory cal !!!!
    mem_size_fm_final = final_counters * ((1 / (fm_final_epsilon ** 2)) * (math.log(1 / delta)))

    # answer 4.b - print the estimators graph
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('COPIES')
    ax1.set_ylabel('morris', color=color)
    ax1.plot(list(beta_counters), morris_beta_res, color=color)
    plt.axhline(y=len(stream), color=color, linestyle='--') #accuracy
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('FM', color=color)  # we already handled the x-label with ax1
    ax2.plot(list(beta_counters), fm_beta_res, color=color)
    plt.axhline(y=1000, color='blue', linestyle='--') #accuracy
    ax2.tick_params(axis='y', labelcolor=color)

    ax3 = ax2.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:green'
    ax3.set_ylabel('FM_final', color=color)  # we already handled the x-label with ax1
    ax3.plot(list(range(0, len(final_counters))), fm_final_res, color=color)
    plt.axhline(y=1000, color='green', linestyle='--') #accuracy
    ax3.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    # memory size graph
    mem_fig, mem_ax1 = plt.subplots()

    color = 'tab:red'
    mem_ax1.set_xlabel('COPIES')
    mem_ax1.set_ylabel('morris memory', color=color)
    mem_ax1.plot(range(len(beta_counters)), mem_size_morris, color=color)
    mem_ax1.tick_params(axis='y', labelcolor=color)

    mem_ax2 = mem_ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    mem_ax2.set_ylabel('FM beta memory', color=color)  # we already handled the x-label with ax1
    mem_ax2.plot(range(len(beta_counters)), mem_size_fm_beta, color=color)
    mem_ax2.tick_params(axis='y', labelcolor=color)

    mem_ax3 = mem_ax2.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:green'
    mem_ax3.set_ylabel('FM_final memory', color=color)  # we already handled the x-label with ax1
    mem_ax3.plot(range(len(final_counters)), mem_size_fm_final, color=color)
    mem_ax3.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    # normalized var graph
    var_fig, var_ax1 = plt.subplots()

    color = 'tab:red'
    var_ax1.set_xlabel('COPIES')
    var_ax1.set_ylabel('morris var', color=color)
    var_ax1.plot(range(len(beta_counters)), var_morris_beta, color=color)
    var_ax1.tick_params(axis='y', labelcolor=color)

    var_ax2 = var_ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    var_ax2.set_ylabel('FM beta memory', color=color)  # we already handled the x-label with ax1
    var_ax2.plot(range(len(beta_counters)), var_fm_beta, color=color)
    var_ax2.tick_params(axis='y', labelcolor=color)

    var_ax3 = var_ax2.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:green'
    var_ax3.set_ylabel('FM_final memory', color=color)  # we already handled the x-label with ax1
    var_ax3.plot(range(len(final_counters)), var_fm_final, color=color)
    var_ax3.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


