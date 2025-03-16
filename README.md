# Reinforcement-Learning On Policy n-Step TD algorithm &amp; Off Policy Monte-Carlo prediction algorithm
# **On-Policy n-Step TD & Off-Policy Monte Carlo Prediction**

## **Introduction**
This repository explores **On-Policy n-Step Temporal Difference (TD) Learning** and **Off-Policy Monte Carlo (MC) Prediction** algorithms in the context of reinforcement learning (RL). These methods are widely used for **value function estimation** in Markov Decision Processes (MDPs).

- **On-Policy n-Step TD**: A balance between Monte Carlo (MC) and TD(0), leveraging **bootstrapping** and multi-step returns.
- **Off-Policy Monte Carlo Prediction**: Uses **importance sampling** to learn from a different behavior policy than the target policy.

---

## ** Algorithms Overview**
### **1️ On-Policy n-Step TD Learning**
- Learns the **value function** \( V(s) \) by updating based on an **n-step return** rather than a single-step TD error.
- Works with an **epsilon-greedy** policy or other soft policies.
- **Combines the advantages** of TD(0) and Monte Carlo methods.

 **Update Rule:**
```math
V(s_t) \leftarrow V(s_t) + \alpha \left( G_t - V(s_t) \right)
```
where the return is:
$$
G_t = R_t + \gamma R_{t+1} + \dots + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n})
$$

### ** Off-Policy Monte Carlo Prediction**
- Estimates the value function of a **target policy** \( \pi \) while following a **behavior policy** \( b \).
- Uses **importance sampling** to correct the discrepancy between the policies.

 **Importance Sampling Ratio:**
$$
\frac{\pi(a|s)}{b(a|s)}
$$
- **Ordinary Importance Sampling (Unbiased, High Variance)**
- **Weighted Importance Sampling (Biased, Lower Variance)**

---

## **Implementation**
- The repository includes **Python implementations** for both algorithms.
- Supports different **discount factors**, **step sizes**, and **epsilon-greedy exploration
