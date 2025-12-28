import numpy as np
import Verification_CartPole_Unknown_Policy
#import matplotlib.pyplot as plt
import torch.optim as optim
import torch
from torch import nn
from dreal import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



# Neural network controller definition
class controller(nn.Module):
    def __init__(self):
        super(controller, self).__init__()
        self.lay1 = nn.Linear(2, 8).double()
        self.lay2 = nn.Linear(8, 1).double()

    def forward(self, u):
        x1 = torch.sigmoid(self.lay1(u))
        hard_tanh = nn.Hardtanh(-80, 80)
        y = hard_tanh(self.lay2(x1))
        return y
        
    
    
# Control Barrier Function (CBF) definition
def CBF(coeff, theta, omega):
    
    polynomial = coeff[0] * omega**2 + coeff[1] * theta**2 + coeff[2] * omega * theta + coeff[3] * omega + coeff[4] * theta + 1

    return polynomial

# CBF evaluated at the next state
def CBF_next(coeff, theta, omega):
    
    theta_plus, omega_plus = dync(theta, omega, Ts)


    polynomial = coeff[0] * omega_plus**2 + coeff[1] * theta_plus**2 + coeff[2] * omega_plus * theta_plus  + coeff[3] * omega_plus + coeff[4] * theta_plus + 1 

    return polynomial
    

# Loss for unsafe states (CBF should be negative)
def Loss_unsafe_fcn(coeff, theta, omega):
    return torch.sum(  torch.relu(CBF(coeff, theta, omega))  )


# Penalize violations of the CBF constraint: h(x⁺) - (1-gamma)h(x) ≥  - L·d_max

def Loss_CBF_fcn(coeff, theta, omega, gamma, lip, d_max):
    cbf_value = CBF(coeff, theta, omega)
    cbf_next_value = CBF_next(coeff, theta, omega) - (1-gamma)*cbf_value
    cbf_const = cbf_next_value - lip*d_max
    return torch.sum(torch.relu( (-1 / (1 + torch.exp(-2*cbf_value)) + 0.45) * (cbf_const) ) )


# Total loss function
def Loss_fcn(coeff, unsafe_th_data, unsafe_om_data, safe_th_data, safe_om_data, gamma, lip, d_max):

    w1, w2, w3, w4, w5, w6 = 1, 1, 1, 1, 1, 1
    loss_unsafe  = w1*Loss_unsafe_fcn(coeff, unsafe_th_data, unsafe_om_data)
    
    
    L_e = coeff[2]**2 - 4*coeff[1]*coeff[0]
    loss_L = w2 * torch.relu ( L_e )
    
    Delta = coeff[1]*coeff[0] + 0.25*coeff[2]*coeff[4]*coeff[3] - 0.25*(coeff[1]*coeff[3]**2 + coeff[0]*coeff[4]**2 + coeff[2]**2)
    loss_Delta = w3* torch.relu ( coeff[0]*Delta )
    
    
    area_denominator = (L_e)**2 if L_e < 0 else 10  # Avoid division by zero
    
    area = -2 * 3.14 * (coeff[1]*coeff[3]**2 + coeff[0]*coeff[4]**2 - coeff[2]*coeff[4]*coeff[3] + L_e) * ( torch.sqrt(- L_e) ) /area_denominator
    
    loss_elipse = w4 * torch.relu( -area + 1 )
    
    
    loss_cbf = w5*Loss_CBF_fcn(coeff, safe_th_data, safe_om_data, gamma, lip, d_max)
    
    
    loss_lip = w6 * lipschitz_penalty(coeff, safe_th_data, safe_om_data, lip)


    return loss_unsafe + loss_L + loss_Delta + loss_elipse +  loss_cbf + loss_lip 


# System dynamics (discrete-time)
def dync(theta, omega, Ts):

    input_ctrl = torch.stack([theta, omega], dim=1)    
    #x_plus = v*Ts + x
    #v_plus = v + (Ts/(mc + mp*torch.sin(theta)**2))*( ctrl_1(input_ctrl).squeeze(1) + mp*torch.sin(theta)*(l*omega**2 - 9.81*torch.cos(theta)) )
    theta_plus = omega*Ts + theta
    omega_plus = omega + (Ts/(l*(mc + mp*torch.sin(theta)**2)))*( -ctrl_1(input_ctrl).squeeze(1)*torch.cos(theta) - mp*l*omega**2*torch.cos(theta)*torch.sin(theta) + (mc + mp)*9.81*torch.sin(theta) )
    return theta_plus, omega_plus


# Lipschitz continuity penalty for the CBF
def lipschitz_penalty(coeff, theta, omega, L):
    theta = theta.requires_grad_(True)
    omega = omega.requires_grad_(True)

    y = coeff[0] * omega**2 + coeff[1] * theta**2 + coeff[2] *  theta * omega + coeff[3] * omega + coeff[4] * theta + 1
    
    grad_outputs = torch.ones_like(y)
    gradients = torch.autograd.grad(outputs=y, inputs=[theta, omega], grad_outputs=grad_outputs,
                                    create_graph=True, retain_graph=True, only_inputs=True)

    grad_x1, grad_x2 = gradients
    grad_norm = torch.sqrt(grad_x1**2 + grad_x2**2)  # Compute ||∇f(x)||

    cbf_value = CBF(coeff, theta, omega)

    penalty = torch.sum( torch.relu( torch.relu(cbf_value + 0.001)* ((grad_norm - L)) ) )  # Penalize if greater than L

    
    return penalty







####

# System parameters
Ts = 0.05          # Sampling time (s)
gamma = 1          # CBF/constraint scaling factor
mc = 2             # Cart mass
mp = 0.1           # Pole mass
l = 1              # Pole length
r_safe = np.pi/4   # Radius of the safe set in (theta, omega) space

# Lipschitz-related constants
lip = 4            # Lipschitz constant bound for the CBF
d_max = 0.001      # Maximum one-step state variation (robustness margin)

# Initialize neural network controller
ctrl_1 = controller()
ctrl_1 = ctrl_1.to(device)


# =========================
# Generate UNSAFE data
# =========================

n_unsafe = 5
theta = np.linspace(0, 2 * np.pi, n_unsafe)

temp_1 = (r_safe + 1e-3) * np.cos(theta)
temp_2 = (r_safe + 1e-3) * np.sin(theta)

unsafe_th_data = torch.tensor(temp_1.T, dtype=torch.double, device = device)
unsafe_om_data = torch.tensor(temp_2.T, dtype=torch.double, device = device)


unsafe_th_data = unsafe_th_data.to(device)
unsafe_om_data = unsafe_om_data.to(device)


# =========================
# Generate SAFE data
# =========================

n_safe = 10

theta_np = 2 * np.pi * np.random.rand(n_safe)
r_np = r_safe * np.sqrt(np.random.rand(n_safe))
x_data = r_np * np.cos(theta_np)
y_data = r_np * np.sin(theta_np)

safe_th_data = torch.tensor(x_data, dtype=torch.double, device=device)
safe_om_data = torch.tensor(y_data, dtype=torch.double, device=device)


safe_th_data = safe_th_data.to(device)
safe_om_data = safe_om_data.to(device)


# Initialize CBF polynomial coefficients
# h(theta, omega) = A*omega^2 + B*theta^2 + C*theta*omega
#                   + D*omega + E*theta + 1

coeff = torch.tensor(
    [-1.9323, -2.4500, -0.3966,  0.0029, -0.3065],
    dtype=torch.double,
    requires_grad=True,
    device=device)

# Optimizer jointly updating:
#   - CBF coefficients
#   - Neural network controller parameters
optimizer = optim.SGD([coeff, *ctrl_1.parameters()], lr= 0.4, momentum=0)

# Flag indicating whether the CBF has been successfully verified
flag_verified = 0

# Tracks the number of iterations after reaching the maximum epoch
# where optimization is not reset and the loss has not converged to zero
iteration_reset = 0

# Total number of optimization iterations performed
overal_iteration = 0

while not flag_verified:
    overal_iteration += 1
    print("========================================= Overal Iteration = ", overal_iteration)
    iteration_reset += 1

    epoch = 0
    flag_loss = 0 # Flag set to 1 when the loss becomes zero
    flag_nan  = 0
    
    batch_size = 250
    num_batches_unsafe = max(1, int(np.ceil(unsafe_th_data.shape[0] / batch_size)))
    num_batches_safe = max(1, int(np.ceil(safe_th_data.shape[0] / batch_size)))

    # Use the larger number of batches to ensure all samples are covered
    num_batches = max(num_batches_unsafe, num_batches_safe)
    print("num_batches   =  ", num_batches)

    while epoch < 5000 and not flag_loss:
        epoch += 1
        epoch_loss = 0

        indices_unsafe = torch.randperm(unsafe_th_data.shape[0])
        indices_safe = torch.randperm(safe_th_data.shape[0])


        
        for batch_idx in range(num_batches):
            

            unsafe_start = batch_idx * batch_size
            safe_start = batch_idx * batch_size

            batch_unsafe_indices = indices_unsafe[unsafe_start : min(unsafe_start + batch_size, unsafe_th_data.shape[0])]
            batch_safe_indices = indices_safe[safe_start : min(safe_start + batch_size, safe_th_data.shape[0])]


            batch_unsafe_th = unsafe_th_data[batch_unsafe_indices]
            batch_unsafe_om = unsafe_om_data[batch_unsafe_indices]
            
            batch_safe_th = safe_th_data[batch_safe_indices]
            batch_safe_om = safe_om_data[batch_safe_indices]
            
            optimizer.zero_grad()  
            
            batch_loss = Loss_fcn(coeff,batch_unsafe_th, batch_unsafe_om, batch_safe_th, batch_safe_om, gamma, lip, d_max)

            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()  

        if math.isnan(epoch_loss):
            flag_nan = 1
            print("loss  =  ", epoch, epoch_loss)

        if epoch % 10 == 0:
            print("loss  =  ", epoch, epoch_loss)
        if epoch_loss == 0: #and epoch > 00:
            flag_loss = 1
            print(epoch, epoch_loss)

    print('coeff = ', coeff)

    if flag_nan:
        iteration_reset = 0 
        print("~~~~~~~~~~~~~nan reset optimizer~~~~~~~~~~!!!!")
        coeff = (8 * torch.rand(5, dtype=torch.double, device=device) - 4).requires_grad_()
        
        ctrl_1 = controller()
        ctrl_1 = ctrl_1.to(device)
        optimizer = optim.SGD([coeff, *ctrl_1.parameters()], lr= 0.01, momentum=0)
    
    if flag_loss:
        print("!!!!!!!!!!!!! = ", Loss_fcn(coeff, unsafe_th_data, unsafe_om_data, safe_th_data, safe_om_data, gamma, lip, d_max) )
        iteration_reset = 0

# Safety Check

        coeff_py = coeff.detach().cpu().numpy()  # Convert PyTorch tensor to NumPy array
        coeff_py = coeff_py.tolist()  # Convert NumPy array to a standard Python list

        
        theta = Variable("theta")
        omega = Variable("omega")
        
        h = coeff_py[0] * omega**2 + coeff_py[1] * theta**2 + coeff_py[2] * omega * theta + coeff_py[3] * omega + coeff_py[4] * theta + 1
        
        f_sat = And( -(r_safe + 1) <= theta, theta <= (r_safe + 1), -(r_safe + 1) <= omega, omega <= (r_safe + 1), 1*1e-4 <= h, - theta**2  - omega**2 + (r_safe)**2 < -1.01*1e-4)
        result = CheckSatisfiability(f_sat, 1e-4)
        print("first safety checked finished !")
        
        if (result is not None):
            print("safe counter example is found  =  ", result)

            x_ce_th_tens = torch.tensor([(result[theta].lb() + result[theta].ub())/2], dtype=torch.double, device = device)
            x_ce_om_tens = torch.tensor([(result[omega].lb() + result[omega].ub())/2], dtype=torch.double, device = device)
            
            unsafe_th_data = torch.cat((unsafe_th_data, x_ce_th_tens))
            unsafe_om_data = torch.cat((unsafe_om_data, x_ce_om_tens))
                        

            CC = coeff_py[0] * x_ce_om_tens**2 + coeff_py[1] * x_ce_th_tens**2 + coeff_py[2] * x_ce_om_tens * x_ce_th_tens + coeff_py[3]* x_ce_om_tens + coeff_py[4]* x_ce_th_tens + 1
               
            print("~~~~~ h(x)  =   ", CC, "~~~~~ s(x)  =   ", -x_ce_th_tens**2 - x_ce_om_tens**2 + (r_safe)**2)
        else:
        
            flag_verified, x_ce = Verification_CartPole_Unknown_Policy.verification_fcn(coeff_py[0], coeff_py[1], coeff_py[2], coeff_py[3], coeff_py[4], 1, Ts, gamma, lip, d_max)
            
            if not flag_verified:
                print("counterexample ?= ", x_ce)

                x_ce_th_tens = torch.tensor([x_ce[0]], dtype=torch.double, device = device)
                x_ce_om_tens = torch.tensor([x_ce[1]], dtype=torch.double, device = device)
                
                safe_th_data = torch.cat((safe_th_data, x_ce_th_tens))
                safe_om_data = torch.cat((safe_om_data, x_ce_om_tens))


                cbf_value_temp = CBF(coeff, x_ce_th_tens, x_ce_om_tens)
                cbf_next_value_temp = CBF_next(coeff, x_ce_th_tens, x_ce_om_tens)

                loss_value_temp = Loss_fcn(coeff, unsafe_th_data, unsafe_om_data, x_ce_th_tens, x_ce_om_tens, gamma, lip, d_max)
                
                print("~~~~~~~~~loss~~~ = ",  loss_value_temp )
            else:
                print("hoooooraaay verified +++++++++++ = ", coeff)

    if iteration_reset >= 3:
        iteration_reset = 0 
        print("reset optimizer!!!!")
        coeff = torch.tensor([-1,  -1,   -1, -1, 0], dtype=torch.double, requires_grad=True, device = device)
        ctrl_1 = controller()
        ctrl_1 = ctrl_1.to(device)
        optimizer = optim.SGD([coeff, *ctrl_1.parameters()], lr= 0.2, momentum=0)
