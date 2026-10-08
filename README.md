# Symbolic regression with a neural network, with and without a PINN

Give a network a list of numbers, and it writes back the formula that made them. For example, it looks at 100 points from a hidden curve and answers `x*sin(2*x)`.

The notebook has two parts:

- **Part A (no PINN):** the network gets 100 clean points and predicts the formula.
- **Part B (with PINN):** the network only gets 12 noisy points. A small physics-informed neural network (PINN) first rebuilds the curve using a known derivative law, and then the formula is read from the rebuilt curve.

Most of this README is about Part B, since that is the new part.

## Contents

1. [How Part A works](#how-part-a-works)
2. [The PINN in Part B](#the-pinn-in-part-b)
3. [Results](#results)
4. [Limits](#limits)
5. [Running it](#running-it)
6. [Background and credits](#background-and-credits)

## How Part A works

1. **Make random formulas.** Formulas are stored as trees using `+`, `-`, `*`, `abs`, `sin`, `tan` and `exp`. SymPy turns each tree into an expression and computes its values at 100 points between -1 and 1.
2. **Turn formulas into tokens.** Each tree is flattened into a list of integers, so `x*sin(x+x)` becomes `*, x, sin, +, x, x, None`.
3. **Train.** The network sees the data points plus the tokens written so far, and learns to guess the next token.
4. **Write a formula.** At test time it starts from an empty sequence and adds one token at a time until it writes an end token.

## The PINN in Part B

### The situation

Imagine you measured a position f(x) at only 12 places, and the readings are noisy. The two obvious fixes are poor. Joining the dots with straight lines follows the noise. Fitting a plain neural network gives a smooth curve, but it can wiggle however it likes between the points.

A PINN adds one more source of information: a law the true curve must obey.

### The equation used

The law in this notebook is a first-order ordinary differential equation with a known right-hand side:

$$\frac{dy}{dx} = s(x), \qquad x \in [-1, 1]$$

Here $y(x)$ is the unknown curve we want, and $s(x)$ is a known function. Think of $y$ as position and $s$ as a known velocity law.

In the demo, $s(x)$ is the derivative of the true formula, computed with SymPy. For example, if the hidden curve is $y = x\sin(2x)$, then $s(x) = \sin(2x) + 2x\cos(2x)$. The network never sees the formula for $y$, only the 12 noisy values of $y$ and the values of $s$.

### Why this pins the curve down

Integrate both sides from $-1$ to $x$:

$$y(x) = y(-1) + \int_{-1}^{x} s(t)\,dt = C + S(x)$$

where $S(x)$ is a known function and $C$ is a single unknown number. So the law fixes the whole *shape* of the curve and leaves only one number free: its vertical offset. The 12 noisy measurements only have to find that one number, and averaging 12 readings cancels most of the noise.

A rough check: with noise standard deviation $\sigma = 0.02$ and 12 points, the offset should be off by about $\sigma/\sqrt{12} \approx 0.006$. The measured average error of the PINN curve was 0.005, which is in the same range.

### The network

A small fully connected network $u_\theta(x)$ with three hidden layers of 32 units and **tanh** activations, one input and one output. Tanh is used because it is smooth. A ReLU network has a flat, piecewise-constant derivative, which is no use for matching a derivative.

### The loss

Training minimizes two terms:

$$\mathcal{L}(\theta) = \underbrace{\frac{1}{N_d}\sum_{i=1}^{N_d}\big(u_\theta(x_i) - y_i\big)^2}_{\text{fit the measurements}} \;+\; w\;\underbrace{\frac{1}{N_c}\sum_{j=1}^{N_c}\left(\frac{du_\theta}{dx}(x_j) - s(x_j)\right)^2}_{\text{obey the law}}$$

- $N_d = 12$ noisy measurements $(x_i, y_i)$
- $N_c = 100$ collocation points $x_j$, spread evenly over $[-1, 1]$. These are points where we check the law. No measurements are needed there.
- $w$ is the physics weight (`phys_weight`, set to 1.0). With $w = 0$ you get the plain network fit, which is the control in the results table.

The second term is called the residual. If the network obeyed the law perfectly, the residual would be zero everywhere.

### Autodifferentiation

The residual needs $du_\theta/dx$. We don't approximate it with finite differences (the difference between nearby outputs). Autodifferentiation gives the exact derivative of the network, by applying the chain rule through every layer.

For one hidden layer $h = \tanh(Wx + b)$ and output $u = v^\top h + c$, the derivative is

$$\frac{du}{dx} = v^\top\Big(\big(1 - \tanh^2(Wx + b)\big) \odot W\Big)$$

and with several layers the same rule is simply chained. PyTorch does this for us, by recording the operations and replaying them backwards.

The notebook uses three lines for it:

```python
u = net(xc)
du = torch.autograd.grad(u, xc, torch.ones_like(u), create_graph=True)[0]
loss = loss + phys_weight * ((du - src) ** 2).mean()
```

Two details are worth knowing:

- **`torch.ones_like(u)`.** The network is applied to 100 points at once, and each output depends only on its own input point. Weighting every output by 1 therefore returns the derivative at each point separately.
- **`create_graph=True`.** This is what makes it a PINN. The derivative is itself a function of the network weights $\theta$, and the optimizer needs the gradient of the residual loss with respect to $\theta$. Without this flag, the derivative would be treated as a fixed number and the physics term would teach the network nothing. So training takes a gradient *of a gradient*.

### Training settings

| Setting | Value |
|---|---|
| Optimizer | Adam, learning rate 0.005, cosine decay |
| Steps | 2500 |
| Precision | double (float64) |
| Measurements | 12 random points, Gaussian noise with standard deviation 0.02 |
| Collocation points | 100, even grid on [-1, 1] |
| Physics weight $w$ | 1.0 |

Each test function gets its own freshly trained network. The PINN is not trained once and reused.

### From the rebuilt curve to a formula

After training, the PINN is evaluated on the 100-point grid. Those 100 values go into the Part A network, which was trained on clean data only, and it writes the formula.

## Results

**Part A:** 98% valid, 29% same curve on unseen test formulas.

How the scores are defined:

| Metric | Meaning |
|---|---|
| valid | the predicted tokens make a legal formula |
| same form | the prediction is written exactly like the true formula |
| same curve | the prediction stays within 0.01 of the true curve at every grid point |

"Same form" is strict, because `x + x` and `2*x` are the same function written two ways. "Same curve" is the fairer check.

**Part B** (40 test cases, 12 noisy points each, same trained network every time):

| Input to the network | Error vs true curve | Same curve |
|---|---|---|
| Clean data (best case) | 0.000 | 32.5% |
| Linear interpolation | 0.112 | 10.0% |
| Plain network fit, no physics | 0.057 | 10.0% |
| PINN | 0.005 | 32.5% |

The physics term is what helps. The same small network without it does no better than interpolation, and with it the result matches clean data.

## Limits

- **The law is built from the answer.** In this demo, $s(x)$ comes from the true formula, so the physics is not independent of what we are trying to find. The law is also directly integrable, so you could in principle integrate $s$ yourself. The demo shows the method working. It is not a realistic discovery problem. A better test would use a law like $y' = -ky + g(x)$, where the solution is not just an integral. In the code that only means changing the residual line.
- **The PINN needs known physics.** With no known equation, there is nothing to enforce, and Part A is the right tool.
- **The sample is small.** There are 40 cases and one random seed, so small differences are noise. The gap between 10% and 32.5% is big enough to notice, but it is not a full study.
- **The model is simple.** It is a fully connected network, and I kept the formulas short (1 to 3 binary and 1 to 2 unary operations) so it could learn them. That leaves about 2,000 unique formulas. A transformer and more data would do better.
- **The network was trained on clean data only.** Training it on reconstructed noisy inputs would probably help.
- **The weight $w$ was not tuned.** I used 1.0 and did not try other values.

## Running it

Install the packages:

```
pip install torch sympy numpy matplotlib jupyter
```

Then open `symbolic_regression_without_and_with_pinn.ipynb` and run the cells from top to bottom.

On a CPU it takes about 12 minutes: roughly 8 for training and 4 for the PINN fits. A GPU speeds up the training part. Everything is seeded, so results should be close between runs, though small differences across machines are normal.

You can change these in the notebook:

- `n_trees`, and the ranges in the dataset loop, for the size and difficulty of the formulas
- `n_obs` and `noise_std` for how sparse and noisy the measurements are
- `phys_weight` for how strongly the PINN follows the law
- `n_steps` in `train` for training time

## Background and credits

PINNs fit a network to data and to a differential equation at the same time. The general form is a residual $\mathcal{F}(x, u, u', u'', \dots) = 0$ that is driven to zero at collocation points, usually together with boundary or initial conditions. The idea of training a network on an equation goes back to Lagaris, Likas and Fotiadis (1998), and the modern version was popularized by Raissi, Perdikaris and Karniadakis (2019). This notebook uses the simplest case, a first-order equation in one variable.

- Raissi, M., Perdikaris, P., Karniadakis, G.E. (2019). Physics-informed neural networks: a deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686–707.
- Lagaris, I.E., Likas, A., Fotiadis, D.I. (1998). Artificial neural networks for solving ordinary and partial differential equations. *IEEE Transactions on Neural Networks*, 9(5), 987–1000.

The symbolic regression part builds on the workshop by Dr. Ben Moseley (Imperial College London), which follows the ideas in Lample and Charton (2019) and Kamienny et al. (2022). The PINN part and the code in this notebook are my own additions and rewrite.

## License

Add your license here.
