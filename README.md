
# PHASE 1: COMPLETE ANN PROJECT FOR HOT DEFORMATION BEHAVIOR

**Model Used** : Multilayer Perceptron (MLP) with Backpropagation

## AISI 304 STAINLESS STEEL - FLOW STRESS PREDICTION

> - Project: Artificial Neural Network for Predicting Hot Deformation Behavior 
> 
> - Method: Multilayer Perceptron (MLP) with Backpropagation
> 
> - Activation Functions: tanh (hidden) + linear (output)
> 
> - Optimizer: Adam with Learning Rate = 0.01
> 
> - Loss Function: Mean Squared Error (MSE)

### The Core Problem 
The Project is working with **AISI 304 Stainless Steel** - a metal used in manufacturing (automotive, aerospace, industrial equipment).

**The Real Question:** When we heat this metal and squeeze it (deformation), how much force do we need?

**Why it matters:**
- Engineers need to know: "If I forge at 1000°C with a squeeze speed of 1 s⁻¹, how hard will the metal resist?"
- Getting this wrong = broken metal or wasted time/resources
- The paper shows you CAN predict this with high accuracy

### The Three Factors That Matter

1. **Temperature (T)** 
   - Cold metal = hard to squeeze (high resistance)
   - Hot metal = easy to squeeze (low resistance)
   - Range in study: 950-1050°C

2. **Strain Rate (ε̇)** - How fast you squeeze
   - Slow squeeze = metal can "relax" → lower resistance
   - Fast squeeze = metal can't relax → higher resistance
   - Range in study: 0.1 to 15 times per second

3. **Strain (ε)** - How much you've already squeezed
   - Initial squeezing = gets harder (work hardening)
   - After some squeezing = reaches stable state
   - Range in study: 0% to 70% deformation

### What We're Predicting
**Flow Stress (σ)** - The force per unit area needed to keep deforming the metal

Think of it like:
- Easy to bend a warm rubber band (low flow stress)
- Hard to bend a cold rubber band (high flow stress)
