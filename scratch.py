## SIMULATED ANNEALING
T = 0.30
for t in range(1000):
    pr = fp_decay_schedule.evaluate(t)
    T_t = T * 0.99**t
    pr_1 = math.e ** (-1 / T_t)
    pr_10 = math.e ** (-10 / T_t)
    print(f"{t:3}:{pr:6.6f}   T:{T_t:0.6f}  {pr_1:0.6f} {pr_10:0.6f}")


for T in range(1, 10):
    for d in range(-1, -10, -1):
        pr = math.e ** (d / T)
        print(f"d:{d} T:{T}  {pr:6.3f}")
