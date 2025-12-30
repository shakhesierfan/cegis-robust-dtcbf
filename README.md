# Counterexample-Guided Synthesis of Robust Discrete-Time Control Barrier Functions

This repository accompanies the paper

**Counterexample-Guided Synthesis of Robust Discrete-Time Control Barrier Functions**  
Erfan Shakhesi, Alexander Katriniok, and W. P. M. H. Heemels  
*IEEE Control Systems Letters*, 2025.
DOI: 10.1109/LCSYS.2025.3578971

---

## Overview

This work proposes a **counterexample-guided framework** for synthesizing
**robust discrete-time control barrier functions (R-DTCBFs)**.
The method iteratively alternates between:
- training candidate R-DTCBFs on sampled states, and
- formally verifying their validity over the entire region of interest,
  adding counterexamples when violations are found.

The resulting R-DTCBFs can be used **online** to synthesize safe controllers
under **input constraints** and **bounded disturbances**.

---

## Cart-Pole Example

The repository includes a discretized cart-pole system with bounded control input,
where the safety objective is to keep the pole angle and angular velocity within
a prescribed region:
\[
\theta^2 + \omega^2 \le (\pi/4)^2.
\]
A quadratic R-DTCBF is synthesized using the proposed CEGIS approach and verified
over the entire state space of interest.

---

## Usage

Run `algorithm-Inverted.py` to execute the algorithm.

Dependencies: NumPy, PyTorch, Pyomo, dReal, and AMPL (with the Couenne solver).

For AMPL installation instructions, please refer to [the AMPL documentation](https://dev.ampl.com/ampl/install.html).
# Installation

This project has been tested on **Ubuntu 22.04**.


---

### Python Dependencies

The following Python packages are required:

- `numpy`
- `pyomo`
- `pyinterval`
- `casadi`
- `torch`
- `dreal`
- `amplpy`

---
Installation & Environment Setup
================================

This project has been tested on Ubuntu 22.04.

The following steps describe how to install all required system dependencies,
Python packages, and external solvers.

--------------------------------------------------
System Requirements
--------------------------------------------------

Operating System: Ubuntu 22.04
Python: 3.8 or newer
Package Manager: python3-pip

--------------------------------------------------
1. Update System and Install pip
--------------------------------------------------
```bash
sudo apt update
sudo apt install -y python3-pip
```
--------------------------------------------------
2. Install Required Python Packages
--------------------------------------------------
```bash
python3 -m pip install --upgrade pip
python3 -m pip install numpy
python3 -m pip install pyomo
python3 -m pip install pyinterval
python3 -m pip install casadi
python3 -m pip install torch
```
--------------------------------------------------
3. Install dReal Solver
--------------------------------------------------
```bash
sudo apt-get install -y curl
curl -fsSL https://raw.githubusercontent.com/dreal/dreal4/master/setup/ubuntu/22.04/install.sh | sudo bash
```
--------------------------------------------------
4. Install AMPL
--------------------------------------------------
```bash
python3 -m pip install amplpy --upgrade
python3 -m amplpy.modules install highs gurobi
python3 -m amplpy.modules activate <license-uuid>
```
A free AMPL Community Edition license can be obtained from:
https://ampl.com/ce

--------------------------------------------------
7. Install Couenne Solver (via AMPL)
--------------------------------------------------
```bash
python3 -m amplpy.modules install coin
```

---


## Citation

If you use this code, please cite the paper:

```bibtex
@article{Shakhesi2025CEGISDTCBF,
  journal={IEEE Control Systems Letters}, 
  title={Counterexample-Guided Synthesis of Robust Discrete-Time Control Barrier Functions}, 
  year={2025},
  volume={9},
  number={},
  pages={1574-1579},
  keywords={Training;Safety;Optimization;Vectors;Discrete-time systems;Neural networks;Lyapunov methods;Learning systems;Data mining;Control systems;Control barrier functions;Lyapunov methods;neural networks;robust control},
  doi={10.1109/LCSYS.2025.3578971}
}
