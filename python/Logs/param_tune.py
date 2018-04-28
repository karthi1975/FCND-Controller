
alt_t_rise =0.71
alt_delta = 0.9
alt_omega_n = 1.57 / alt_t_rise
Kp = alt_omega_n ** 2
Kd= 2 * alt_delta * alt_omega_n

print("Kp , Kd = ",Kp, Kd)

