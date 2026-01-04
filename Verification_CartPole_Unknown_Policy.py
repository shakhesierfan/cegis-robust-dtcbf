import numpy as np
from amplpy import AMPL
import copy
from pyomo.environ import *
from interval import interval, inf, imath
import casadi as ca
#import matplotlib.pyplot as plt
#from matplotlib.patches import Circle
#from matplotlib.patches import Rectangle
#from matplotlib.colors import Normalize
#from matplotlib.cm import get_cmap, ScalarMappable
#import matplotlib
import os, sys, contextlib
import time



def verification_fcn(A, B, C, D, E, F, Ts, gamma, lip, d_max):

# --- Step 2 of the Verification Algorithm: optimization problem formulation ---
    def obj_u(x_bar, gamma):
        
        # AMPL model:
        mod = r"""
        param A; param B; param C; param D; param E; param F;
        param Ts; param g; param mc; param mp; param L; param u_max;
        param th0; param om0;
    
        var u >= -u_max <= u_max;
    
        param denom := (mc + mp * (sin(th0))^2);
    
        var th0_plus;
        var om0_plus;
    
        subject to th0_plus_def:
            th0_plus = th0 + om0 * Ts;
    
        subject to om0_plus_def:
            om0_plus = om0 + Ts * ( -u * cos(th0)
                                    - mp * L * (om0^2) * cos(th0) * sin(th0)
                                    + (mc + mp) * g * sin(th0) ) / (L * denom);
    
        minimize obj:
            -( A * (om0_plus^2)
             + B * (th0_plus^2)
             + C * th0_plus * om0_plus
             + D * om0_plus
             + E * th0_plus
             + F );
        """
        # --- Configure and solve AMPL model with Couenne --------------------------
        ampl = AMPL()
        ampl.eval(mod)
        
        # Assign parameters
        ampl.param["A"] = A; ampl.param["B"] = B; ampl.param["C"] = C
        ampl.param["D"] = D; ampl.param["E"] = E; ampl.param["F"] = F
        ampl.param["Ts"] = Ts; ampl.param["g"] = g
        ampl.param["mc"] = mc; ampl.param["mp"] = mp; ampl.param["L"] = L
        ampl.param["u_max"] = u_max
        ampl.param["th0"] = float(x_bar[0])
        ampl.param["om0"] = float(x_bar[1])

        # Solver options (suppressing log output)
        ampl.setOption("solver", "couenne")
        ampl.setOption("solver_msg", 0)
        ampl.setOption("show_stats", 0)
        ampl.setOption("couenne_options",
                       "bonmin.bb_log_interval=0 bonmin.nlp_log_level=0")
        
        # Run solver silently
        with open(os.devnull, "w") as devnull, \
             contextlib.redirect_stdout(devnull), \
             contextlib.redirect_stderr(devnull):
             ampl.solve()
                 
        # Extract optimization results
        u_star = ampl.getVariable("u").value()
        print("u_star =   ", u_star)
        obj_value = ampl.getObjective("obj").value()

        # Evaluate h(x) at x_bar (to check if it may be a counterexample)
        h_0x  = A * x_bar[1]**2 + B * x_bar[0]**2 + C * x_bar[0] * x_bar[1] + D * x_bar[1] + E * x_bar[0] + F
    
        #if (h_0x >= 0) and obj_value > 0:
        #    print("counterexample : ", x_bar)
    
        return obj_value, u_star, (h_0x >= 0)
    
# --- Step 3 of the Verification Algorithm: optimization problem formulation ---
    def obj_h(th_L, th_U, om_L, om_U, u_star, gamma):

        th_cas = ca.MX.sym('th')
        om_cas = ca.MX.sym('om')
        
        th_int = interval[th_L, th_U]
        om_int = interval[om_L, om_U]
        h_x = A * om_cas**2 + B * th_cas**2 + C * th_cas * om_cas + D * om_cas + E * th_cas + F

        th0_plus = th_cas + om_cas * Ts
        om0_plus = om_cas + Ts*(-u_star*ca.cos(th_cas) - mp*L*om_cas**2 * ca.cos(th_cas) * ca.sin(th_cas) + (mc + mp)*g*ca.sin(th_cas))/(L*(mc + mp * ca.sin(th_cas)**2)) 
        h_plus = A * om0_plus**2 + B * th0_plus**2 + C * th0_plus * om0_plus + D * om0_plus + E * th0_plus + F

        
 # --- Computing alpha_H_1 and alpha_H_2 (for convexifying the constraint)        
        # --- Interval bounds for (constant) Hessian entries of h wrt (th, om) ----
        #   H = -∇² h(θ,ω) = [[-∂²h/∂θ², -∂²h/∂θ∂ω],
        #                    [-∂²h/∂ω∂θ, -∂²h/∂ω²]]
        
        int_H_11 = interval[-2*B, -2*B]
        int_H_12 = interval[-C, -C]
        int_H_22 = interval[-2*A, -2*A]
        
        # Compute alpha_H_1 and alpha_H_2 using Scaled Gerschgorin method: λ_i = H_ii − max_j≠i |H_ij| and alpha_i = max(0, -0.5*λ_i),
        lambda_H_1  = int_H_11[0][0] -  ( np.max( [np.abs(int_H_12[0][1]), np.abs(int_H_12[0][0])] ) )      
        alpha_H_1   = np.max([0, -0.5*lambda_H_1])
        
        lambda_H_2  = int_H_22[0][0] -  ( np.max( [np.abs(int_H_12[0][1]), np.abs(int_H_12[0][0])] ) )
        alpha_H_2   = np.max([0, -0.5*lambda_H_2])

    

# --- Computing alpha_F_1 and alpha_F_2 (for convexifying the objective function)    
        
        # --- Interval second derivatives of h(x⁺): F = ∇² h(x⁺) ------------------
        # int_F_11 = ∂²h(x⁺)/∂θ² over the box
        # int_F_12 = ∂²h(x⁺)/∂θ∂ω over the box
        # int_F_22 = ∂²h(x⁺)/∂ω² over the box
        # (obtained symbolically, then interval-substituted)     
        
        int_F_11 = A*(om_int + Ts*(-L*mp*om_int**2*imath.sin(th_int)*imath.cos(th_int) + g*(mc + mp)*imath.sin(th_int) - u_star*imath.cos(th_int))/(L*(mc + mp*imath.sin(th_int)**2)))*(16*Ts*mp**2*(-L*mp*om_int**2*imath.sin(th_int)*imath.cos(th_int) + g*(mc + mp)*imath.sin(th_int) - u_star*imath.cos(th_int))*imath.sin(th_int)**2*imath.cos(th_int)**2/(L*(mc + mp*imath.sin(th_int)**2)**3) + 4*Ts*mp*(-L*mp*om_int**2*imath.sin(th_int)*imath.cos(th_int) + g*(mc + mp)*imath.sin(th_int) - u_star*imath.cos(th_int))*imath.sin(th_int)**2/(L*(mc + mp*imath.sin(th_int)**2)**2) - 4*Ts*mp*(-L*mp*om_int**2*imath.sin(th_int)*imath.cos(th_int) + g*(mc + mp)*imath.sin(th_int) - u_star*imath.cos(th_int))*imath.cos(th_int)**2/(L*(mc + mp*imath.sin(th_int)**2)**2) - 8*Ts*mp*(L*mp*om_int**2*imath.sin(th_int)**2 - L*mp*om_int**2*imath.cos(th_int)**2 + g*(mc + mp)*imath.cos(th_int) + u_star*imath.sin(th_int))*imath.sin(th_int)*imath.cos(th_int)/(L*(mc + mp*imath.sin(th_int)**2)**2) + 2*Ts*(4*L*mp*om_int**2*imath.sin(th_int)*imath.cos(th_int) - g*(mc + mp)*imath.sin(th_int) + u_star*imath.cos(th_int))/(L*(mc + mp*imath.sin(th_int)**2))) + A*(-4*Ts*mp*(-L*mp*om_int**2*imath.sin(th_int)*imath.cos(th_int) + g*(mc + mp)*imath.sin(th_int) - u_star*imath.cos(th_int))*imath.sin(th_int)*imath.cos(th_int)/(L*(mc + mp*imath.sin(th_int)**2)**2) + 2*Ts*(L*mp*om_int**2*imath.sin(th_int)**2 - L*mp*om_int**2*imath.cos(th_int)**2 + g*(mc + mp)*imath.cos(th_int) + u_star*imath.sin(th_int))/(L*(mc + mp*imath.sin(th_int)**2)))*(-2*Ts*mp*(-L*mp*om_int**2*imath.sin(th_int)*imath.cos(th_int) + g*(mc + mp)*imath.sin(th_int) - u_star*imath.cos(th_int))*imath.sin(th_int)*imath.cos(th_int)/(L*(mc + mp*imath.sin(th_int)**2)**2) + Ts*(L*mp*om_int**2*imath.sin(th_int)**2 - L*mp*om_int**2*imath.cos(th_int)**2 + g*(mc + mp)*imath.cos(th_int) + u_star*imath.sin(th_int))/(L*(mc + mp*imath.sin(th_int)**2))) + 2*B + C*(Ts*om_int + th_int)*(8*Ts*mp**2*(-L*mp*om_int**2*imath.sin(th_int)*imath.cos(th_int) + g*(mc + mp)*imath.sin(th_int) - u_star*imath.cos(th_int))*imath.sin(th_int)**2*imath.cos(th_int)**2/(L*(mc + mp*imath.sin(th_int)**2)**3) + 2*Ts*mp*(-L*mp*om_int**2*imath.sin(th_int)*imath.cos(th_int) + g*(mc + mp)*imath.sin(th_int) - u_star*imath.cos(th_int))*imath.sin(th_int)**2/(L*(mc + mp*imath.sin(th_int)**2)**2) - 2*Ts*mp*(-L*mp*om_int**2*imath.sin(th_int)*imath.cos(th_int) + g*(mc + mp)*imath.sin(th_int) - u_star*imath.cos(th_int))*imath.cos(th_int)**2/(L*(mc + mp*imath.sin(th_int)**2)**2) - 4*Ts*mp*(L*mp*om_int**2*imath.sin(th_int)**2 - L*mp*om_int**2*imath.cos(th_int)**2 + g*(mc + mp)*imath.cos(th_int) + u_star*imath.sin(th_int))*imath.sin(th_int)*imath.cos(th_int)/(L*(mc + mp*imath.sin(th_int)**2)**2) + Ts*(4*L*mp*om_int**2*imath.sin(th_int)*imath.cos(th_int) - g*(mc + mp)*imath.sin(th_int) + u_star*imath.cos(th_int))/(L*(mc + mp*imath.sin(th_int)**2))) + 2*C*(-2*Ts*mp*(-L*mp*om_int**2*imath.sin(th_int)*imath.cos(th_int) + g*(mc + mp)*imath.sin(th_int) - u_star*imath.cos(th_int))*imath.sin(th_int)*imath.cos(th_int)/(L*(mc + mp*imath.sin(th_int)**2)**2) + Ts*(L*mp*om_int**2*imath.sin(th_int)**2 - L*mp*om_int**2*imath.cos(th_int)**2 + g*(mc + mp)*imath.cos(th_int) + u_star*imath.sin(th_int))/(L*(mc + mp*imath.sin(th_int)**2))) + D*(8*Ts*mp**2*(-L*mp*om_int**2*imath.sin(th_int)*imath.cos(th_int) + g*(mc + mp)*imath.sin(th_int) - u_star*imath.cos(th_int))*imath.sin(th_int)**2*imath.cos(th_int)**2/(L*(mc + mp*imath.sin(th_int)**2)**3) + 2*Ts*mp*(-L*mp*om_int**2*imath.sin(th_int)*imath.cos(th_int) + g*(mc + mp)*imath.sin(th_int) - u_star*imath.cos(th_int))*imath.sin(th_int)**2/(L*(mc + mp*imath.sin(th_int)**2)**2) - 2*Ts*mp*(-L*mp*om_int**2*imath.sin(th_int)*imath.cos(th_int) + g*(mc + mp)*imath.sin(th_int) - u_star*imath.cos(th_int))*imath.cos(th_int)**2/(L*(mc + mp*imath.sin(th_int)**2)**2) - 4*Ts*mp*(L*mp*om_int**2*imath.sin(th_int)**2 - L*mp*om_int**2*imath.cos(th_int)**2 + g*(mc + mp)*imath.cos(th_int) + u_star*imath.sin(th_int))*imath.sin(th_int)*imath.cos(th_int)/(L*(mc + mp*imath.sin(th_int)**2)**2) + Ts*(4*L*mp*om_int**2*imath.sin(th_int)*imath.cos(th_int) - g*(mc + mp)*imath.sin(th_int) + u_star*imath.cos(th_int))/(L*(mc + mp*imath.sin(th_int)**2)))

        
        int_F_12 = A*(om_int + Ts*(-L*mp*om_int**2*imath.sin(th_int)*imath.cos(th_int) + g*(mc + mp)*imath.sin(th_int) - u_star*imath.cos(th_int))/(L*(mc + mp*imath.sin(th_int)**2)))*(8*Ts*mp**2*om_int*imath.sin(th_int)**2*imath.cos(th_int)**2/(mc + mp*imath.sin(th_int)**2)**2 + 2*Ts*(2*L*mp*om_int*imath.sin(th_int)**2 - 2*L*mp*om_int*imath.cos(th_int)**2)/(L*(mc + mp*imath.sin(th_int)**2))) + A*(-2*Ts*mp*om_int*imath.sin(th_int)*imath.cos(th_int)/(mc + mp*imath.sin(th_int)**2) + 1)*(-4*Ts*mp*(-L*mp*om_int**2*imath.sin(th_int)*imath.cos(th_int) + g*(mc + mp)*imath.sin(th_int) - u_star*imath.cos(th_int))*imath.sin(th_int)*imath.cos(th_int)/(L*(mc + mp*imath.sin(th_int)**2)**2) + 2*Ts*(L*mp*om_int**2*imath.sin(th_int)**2 - L*mp*om_int**2*imath.cos(th_int)**2 + g*(mc + mp)*imath.cos(th_int) + u_star*imath.sin(th_int))/(L*(mc + mp*imath.sin(th_int)**2))) + 2*B*Ts + C*Ts*(-2*Ts*mp*(-L*mp*om_int**2*imath.sin(th_int)*imath.cos(th_int) + g*(mc + mp)*imath.sin(th_int) - u_star*imath.cos(th_int))*imath.sin(th_int)*imath.cos(th_int)/(L*(mc + mp*imath.sin(th_int)**2)**2) + Ts*(L*mp*om_int**2*imath.sin(th_int)**2 - L*mp*om_int**2*imath.cos(th_int)**2 + g*(mc + mp)*imath.cos(th_int) + u_star*imath.sin(th_int))/(L*(mc + mp*imath.sin(th_int)**2))) + C*(Ts*om_int + th_int)*(4*Ts*mp**2*om_int*imath.sin(th_int)**2*imath.cos(th_int)**2/(mc + mp*imath.sin(th_int)**2)**2 + Ts*(2*L*mp*om_int*imath.sin(th_int)**2 - 2*L*mp*om_int*imath.cos(th_int)**2)/(L*(mc + mp*imath.sin(th_int)**2))) + C*(-2*Ts*mp*om_int*imath.sin(th_int)*imath.cos(th_int)/(mc + mp*imath.sin(th_int)**2) + 1) + D*(4*Ts*mp**2*om_int*imath.sin(th_int)**2*imath.cos(th_int)**2/(mc + mp*imath.sin(th_int)**2)**2 + Ts*(2*L*mp*om_int*imath.sin(th_int)**2 - 2*L*mp*om_int*imath.cos(th_int)**2)/(L*(mc + mp*imath.sin(th_int)**2)))


        int_F_22 = -4*A*Ts*mp*(om_int + Ts*(-L*mp*om_int**2*imath.sin(th_int)*imath.cos(th_int) + g*(mc + mp)*imath.sin(th_int) - u_star*imath.cos(th_int))/(L*(mc + mp*imath.sin(th_int)**2)))*imath.sin(th_int)*imath.cos(th_int)/(mc + mp*imath.sin(th_int)**2) + A*(-4*Ts*mp*om_int*imath.sin(th_int)*imath.cos(th_int)/(mc + mp*imath.sin(th_int)**2) + 2)*(-2*Ts*mp*om_int*imath.sin(th_int)*imath.cos(th_int)/(mc + mp*imath.sin(th_int)**2) + 1) + 2*B*Ts**2 - 2*C*Ts*mp*(Ts*om_int + th_int)*imath.sin(th_int)*imath.cos(th_int)/(mc + mp*imath.sin(th_int)**2) + 2*C*Ts*(-2*Ts*mp*om_int*imath.sin(th_int)*imath.cos(th_int)/(mc + mp*imath.sin(th_int)**2) + 1) - 2*D*Ts*mp*imath.sin(th_int)*imath.cos(th_int)/(mc + mp*imath.sin(th_int)**2)

        # Compute alpha_F_1 and alpha_F_2 using Scaled Gerschgorin method
        lambda_F_1  = int_F_11[0][0] -  ( np.max( [np.abs(int_F_12[0][1]), np.abs(int_F_12[0][0])] ) )
        alpha_F_1   = np.max([0, -0.5*lambda_F_1])
        
        lambda_F_2  = int_F_22[0][0] -  ( np.max( [np.abs(int_F_12[0][1]), np.abs(int_F_12[0][0])] ) )
        alpha_F_2   = np.max([0, -0.5*lambda_F_2])


        
        nlp = {
            'x': ca.vertcat(th_cas, om_cas),
            'f': h_plus + alpha_F_1*(th_L - th_cas)*(th_U - th_cas) +  alpha_F_2*(om_L - om_cas)*(om_U - om_cas),
            'g': -h_x   + alpha_H_1*(th_L - th_cas)*(th_U - th_cas) +  alpha_H_2*(om_L - om_cas)*(om_U - om_cas)
        }
        
        opts = {
            'ipopt': {
                'print_level': 0,
                'tol': 1e-12,
                'max_iter': 5000,
                'constr_viol_tol':1e-12
            },
            'print_time': False
        }

        # Set up solver
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        lbg = [-ca.inf]
        ubg = [0]
        lbx = [th_L, om_L]  
        ubx = [th_U, om_U]  

        sol = solver(x0=[(th_L + th_U)/2, (om_L + om_U)/2], lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        x_min = sol['x']
        x_min = np.array(x_min).flatten()
        
        h_x_value   = A * x_min[1]**2 + B * x_min[0]**2 + C * x_min[0] * x_min[1] + D * x_min[1] + E * x_min[0] + F
                                
        stats = solver.stats()
        
        return sol['f'], h_x_value, x_min, stats['return_status'] == 'Solve_Succeeded'



################
    def obj_lip(th_L, th_U, om_L, om_U, lip):
    
        th_cas = ca.MX.sym('th')
        om_cas = ca.MX.sym('om')
        th_int = interval[th_L, th_U]
        om_int = interval[om_L, om_U]
    
    
        #model = ConcreteModel()
        
        #model.x = Var(within=Reals, bounds=(x_L, x_U), initialize=(x_U + x_L)/2 )
        #model.y = Var(within=Reals, bounds=(y_L, y_U), initialize=(y_U + y_L)/2 )
    
        h_x = A * om_cas**2 + B * th_cas**2 + C * th_cas * om_cas + D * om_cas + E * th_cas + F

        grad_om = 2 * A * om_cas + C * th_cas + D
        grad_th = C * om_cas + 2 * B * th_cas + E
        
        
        
        # --- Computing alpha_H_1 and alpha_H_2 (for convexifying the constraint)        
        # --- Interval bounds for (constant) Hessian entries of h wrt (th, om) ----
        #   H = -∇² h(θ,ω) = [[-∂²h/∂θ², -∂²h/∂θ∂ω],
        #                    [-∂²h/∂ω∂θ, -∂²h/∂ω²]]
        
        int_H_11 = interval[-2*B, -2*B]
        int_H_12 = interval[-C, -C]
        int_H_22 = interval[-2*A, -2*A]
        
        lambda_H_1  = int_H_11[0][0] -  ( np.max( [np.abs(int_H_12[0][1]), np.abs(int_H_12[0][0])] ) )      
        alpha_H_1   = np.max([0, -0.5*lambda_H_1])
        
        lambda_H_2  = int_H_22[0][0] -  ( np.max( [np.abs(int_H_12[0][1]), np.abs(int_H_12[0][0])] ) )
        alpha_H_2   = np.max([0, -0.5*lambda_H_2])
        
        
        int_F_11 = interval[-2*(C**2 + 4*B**2), -2*(C**2 + 4*B**2)]
        int_F_12 = interval[-4*C*(A + B), -4*C*(A + B)]
        int_F_21 = int_F_12
        int_F_22 = interval[-2*(4*A**2 + C**2), -2*(4*A**2 + C**2)]
        
        
        lambda_F_1  = int_F_11[0][0] -  ( np.max( [np.abs(int_F_12[0][1]), np.abs(int_F_12[0][0])] ) )
        alpha_F_1   = np.max([0, -0.5*lambda_F_1])
        
        lambda_F_2  = int_F_22[0][0] -  ( np.max( [np.abs(int_F_12[0][1]), np.abs(int_F_12[0][0])] ) )
        alpha_F_2   = np.max([0, -0.5*lambda_F_2])
       
        
        
        
        nlp = {
            'x': ca.vertcat(th_cas, om_cas),
            'f': -(grad_th**2 + grad_om**2) + alpha_F_1*(th_L - th_cas)*(th_U - th_cas) +  alpha_F_2*(om_L - om_cas)*(om_U - om_cas),
            'g': -h_x   + alpha_H_1*(th_L - th_cas)*(th_U - th_cas) +  alpha_H_2*(om_L - om_cas)*(om_U - om_cas)
        }
        
        print("te!!!!!!!!! = ", alpha_F_1*(th_L - (th_U+th_L)/2)*(th_U - (th_U+th_L)/2) +  alpha_F_2*(om_L - (om_U+om_L)/2)*(om_U - (om_U+om_L)/2))

        opts = {
            'ipopt': {
                'print_level': 0,
                'tol': 1e-12,
                'max_iter': 5000,
                'constr_viol_tol':1e-12
            },
            'print_time': False
        }

        # Set up solver
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        lbg = [-ca.inf]
        ubg = [0]
        lbx = [th_L, om_L]  
        ubx = [th_U, om_U]  

        sol = solver(x0=[(th_L + th_U)/2, (om_L + om_U)/2], lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        
        x_min = sol['x']

        #print("x_min =  ", x_min)
        x_min = np.array(x_min).flatten()
        stats = solver.stats()

        #print("obj_lip_lower", solve.problem.lower_bound, "obj_lip  =  ", model.obj())
        
        return -sol['f'] - lip**2, x_min, stats['return_status'] == 'Solve_Succeeded'



    print("coeff  =  ", A, B, C, D, E)

# --- Cart–pole parameters -----------------------------------------------------    
    mc, mp, L, g = 2, 0.1, 1, 9.81  
    u_max = 80
  
    
# --- Initial domain X^(0) -----------------------------------------------------    
    th_LBD, th_UBD  = -np.pi/4, np.pi/4
    om_LBD , om_UBD = -np.pi/4, np.pi/4
    
        
    finished = 0  
    iteration = 0
    
    rec_pos_th = []
    rec_pos_om = []
    
    
    counter_storing = 1
    
# --- Domain queue (to be verified/branched) -----------------------------------
    rec_pos_th.append([th_LBD, th_UBD])
    rec_pos_om.append([om_LBD, om_UBD])
   
    successful = 0
    x_cexample = []

# --- Branch-and-verify loop ---------------------------------------------------
    while finished == 0:
        
        iteration += 1
        print("iteration = ", iteration)
        
        # Current domain (last-in)
        current_index = len(rec_pos_th) - 1
        current_index_deleted = 0
        
        # Details of the selected domain               
        th_L, th_U = rec_pos_th[current_index]
        om_L, om_U = rec_pos_om[current_index]

        # Stopping criterion (domain diameter)
        size = np.sqrt( (th_U - th_L)**2 + (om_U - om_L)**2 )

        # Step 1 (Verification Algorithm): selected state = domain center
        x_bar = [(th_U + th_L) / 2, (om_U + om_L) / 2]

        # Step 2 (Verification Algorithm): computing the control input at x_bar
        obj_u_star, u_star, flag_u = obj_u(x_bar, gamma)


        if flag_u and (obj_u_star > 0):
        # Counterexample found
            finished = 1
            successful = 0
            x_cexample = [x_bar[0], x_bar[1]]
            print("counterexample is: ", x_bar)
        else:
            # Step 3 (Verification Algorithm): domain verification using the fixed u_star
            obj_h_value, obj_cons_value, x_min, flag_h = obj_h(th_L, th_U, om_L, om_U, u_star, gamma)
            obj_lip_value, x_lip_min, flag_lip         = obj_lip(th_L, th_U, om_L, om_U, lip)
            
        # If domain lies outside C, flag_h = 0; otherwise flag_h = 1
            #print("obj_h_value   = ", obj_h_value)
            print("obj_lip_value =  ", obj_lip_value)
            print("obj_h_value - lip*d_max >? 0 ", obj_h_value - lip*d_max)
            if ( (obj_h_value >= lip*d_max) or (flag_h == 0) ) and (obj_lip_value <= 0 or flag_lip == 0):

            # Verified domain → draw and remove
                #if obj_h_value >= 0 and (flag_h == 1):
                    #draw_rect_colored(th_L, th_U, om_L, om_U, u_star) # Verified

                current_index_deleted = 1    
                del rec_pos_th[current_index]
                del rec_pos_om[current_index]
            else:
            # Not verified yet: check minimum size
                if size < 1e-3:
                    finished = 1
                    successful = 0
                    if  obj_lip_value > 0:
                        x_cexample = x_lip_min
                        print("Region too small; Lipschitz condition violated!!!")
                    else:
                        x_cexample = x_min
                        print("Region too small; returning the center point as a potential counterexample!!!")
                    
            # All domains processed?
            if not rec_pos_th and not finished:
                print("Hoooooorayy!!!!!!!!!!!!!!!")
                finished = 1
                successful = 1
            
        # Branching (bisect along the longest side)
        if current_index_deleted == 0:

            lst = [(th_U - th_L), (om_U - om_L)]
               
            max_value = max(lst)
            max_index = lst.index(max_value)


            if  max_index == 0:
                # Split θ-interval
                rec_pos_th[current_index] = [th_L, (th_U + th_L)/2]
                rec_pos_th.append( [(th_U + th_L)/2, th_U] )
                rec_pos_om[current_index] = [om_L, om_U]
                rec_pos_om.append( [om_L, om_U] )
                       
            elif max_index == 1:
                # Split ω-interval
                rec_pos_th[current_index] = [th_L, th_U]
                rec_pos_th.append( [th_L, th_U] )
                rec_pos_om[current_index] = [om_L, (om_U + om_L)/2]
                rec_pos_om.append( [(om_U + om_L)/2, om_U] )
 
    return successful, x_cexample
    
