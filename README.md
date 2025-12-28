# Counterexample-Guided Synthesis of Robust Discrete-Time Control Barrier Functions

This repository accompanies the paper

**Counterexample-Guided Synthesis of Robust Discrete-Time Control Barrier Functions**  
Erfan Shakhesi, Alexander Katriniok, and W. P. M. H. Heemels  
*IEEE Control Systems Letters (to appear)* :contentReference[oaicite:0]{index=0}

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


## Citation

If you use this code, please cite the paper:

```bibtex
@article{Shakhesi2025CEGISDTCBF,
  title={Counterexample-Guided Synthesis of Robust Discrete-Time Control Barrier Functions},
  author={Shakhesi, Erfan and Katriniok, Alexander and Heemels, W. P. M. H.},
  journal={IEEE Control Systems Letters},
  year={2025}
}


