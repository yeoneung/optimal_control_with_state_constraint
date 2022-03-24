# Optimal control with state constraintd

This documents provide Python code for the experiments conducted in the paper. 

The code is written based on Python 3.8.8 and CVXPY module is needed.

## 1. Run 'lagrangian_perspective.py'

This file commputes optimal control and the state trajectories generated based on our new algorithm.

Images of plots will be saved in the current directory. 

The control is saved as 'control.png' and trajectories are saved as 'x_i.png' for i=1,2,3,4.

## 2. Run comparison_lee_c.py'

This file computes optimal control and the state trajectories generated based on

A Computationally Efficient Hamilton-Jacobi-based Formula for State-Constrained Optimal Control Problems
Donggun Lee, Claire J. Tomlin
https://arxiv.org/abs/2106.13440

Images of plots will be saved in the current directory.

The control is saved as 'control_.png' and trajectories are saved as 'x_i_.png' for i=1,2,3,4.
