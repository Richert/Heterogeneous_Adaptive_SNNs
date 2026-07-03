from numpy import pi, sqrt
from numpy import dot


def vector_field(t,y,dy,tau,Delta,J,eta,tau_s,kappa,tau_a,tau_p,tau_d,b,I_ext_timed_input,I_ext_input,source_idx_v1,source_idx_v2,source_idx_v3,source_idx_v4,source_idx_v5,a_p,a_d,weight,source_idx):


	r = y[0:10]
	v = y[10:20]
	s = y[20:30]
	a = y[30:40]
	u_p = y[40:50]
	u_d = y[50:60]
	w = y[60:160]
	I_ext_timed_input[0] = I_ext_input[t]
	s_in_v1 = s[source_idx_v1]
	p1 = s[source_idx_v2]
	p2 = u_p[source_idx_v3]
	d1 = u_d[source_idx_v4]
	d2 = r[source_idx_v5]
	ltp = a_p*p1*p2
	ltd = a_d*d1*d2
	s_out = s_in_v1*w
	s_in = dot(weight, s_out)
	I_ext = I_ext_timed_input[source_idx]
	
	dy[0:10] = (Delta/(pi*tau) + 2.0*r*v)/tau
	dy[10:20] = (I_ext + J*s_in*tau + eta - pi**2*r**2*tau**2 + v**2)/tau
	dy[20:30] = (a*r - s)/tau_s
	dy[30:40] = kappa*r*(1 - a) + (-a + kappa)/tau_a
	dy[40:50] = (r - u_p)/tau_p
	dy[50:60] = (r - u_d)/tau_d
	dy[60:160] = b*(-ltd*w + ltp*(1 - w)) + (1 - b)*(-ltd + ltp)*(-w**2 + w)

	return dy