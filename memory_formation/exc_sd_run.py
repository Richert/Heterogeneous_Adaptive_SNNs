from numpy import pi, sqrt
from numpy import dot


def exc_sd_vectorfield(t,y,dy,tau,Delta,eta,tau_s,tau_a,kappa,tau_p,tau_d,tau_v1,Delta_v1,eta_v1,tau_s_v1,b,weight_in1,source_idx_in2,weight_in2,weight,I_ext_input,source_idx_v2,source_idx_v3,source_idx_v4,source_idx_v5,source_idx_v6,a_p,a_d,weight_in0,source_idx,source_idx_v1,weight_v1):


	r = y[0:40]
	v = y[40:80]
	s_in1 = y[80:120]
	a = y[120:160]
	u_p = y[160:200]
	u_d = y[200:240]
	r_v1 = y[240:244]
	v_v1 = y[244:248]
	s_in2 = y[248:252]
	w = y[252:1452]
	s_in_in1 = dot(weight_in1, s_in1)
	s_in_in2 = weight_in2*s_in2[source_idx_in2]
	s_in_v1 = dot(weight, s_in1)
	I_ext_timed_input = I_ext_input[t]
	s_in_v2 = s_in1[source_idx_v2]
	p1 = s_in1[source_idx_v3]
	p2 = u_p[source_idx_v4]
	d1 = s_in1[source_idx_v5]
	d2 = u_d[source_idx_v6]
	ltp = a_p*p1*p2
	ltd = a_d*d1*d2
	s_out_in0 = s_in_v2*w
	s_in_in0 = dot(weight_in0, s_out_in0)
	s_in = s_in_in0 + s_in_in1 + s_in_in2
	I_ext = I_ext_timed_input[source_idx]
	I_ext_v1 = dot(weight_v1, I_ext_timed_input[source_idx_v1])
	
	dy[0:40] = (Delta/(pi*tau) + 2.0*r*v)/tau
	dy[40:80] = (I_ext + eta - pi**2*r**2*tau**2 + s_in*tau + v**2)/tau
	dy[80:120] = a*r - s_in1/tau_s
	dy[120:160] = -a*kappa*r + (1 - a)/tau_a
	dy[160:200] = r - u_p/tau_p
	dy[200:240] = r - u_d/tau_d
	dy[240:244] = (Delta_v1/(pi*tau_v1) + 2.0*r_v1*v_v1)/tau_v1
	dy[244:248] = (I_ext_v1 + eta_v1 - pi**2*r_v1**2*tau_v1**2 + s_in_v1*tau_v1 + v_v1**2)/tau_v1
	dy[248:252] = r_v1 - s_in2/tau_s_v1
	dy[252:1452] = b*(-ltd*w + ltp*(1 - w)) + (1 - b)*(-ltd + ltp)*(-w**2 + w)

	return dy