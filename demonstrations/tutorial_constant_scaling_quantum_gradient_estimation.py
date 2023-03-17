r"""
Gradient Estimation with Constant Scaling for Hybrid Models
===========================================================

.. meta::
    :property=“og:description”: Implement a quantum hybrid model optimized using SPSA gradients
    :property=“og:image”: https://pennylane.ai/qml/_images/spsa_mntn.png

*Author: Thomas Hoffmann — Posted: 11 January 2023. Last updated: 11 January 2023.*

In this tutorial, we show how to use the gradient method from SPSA to optimize hybrid quantum-classical neural
networks. Research results are published in `Gradient Estimation with Constant Scaling for Hybrid Quantum Machine Learning
<https://arxiv.org/abs/2211.13981>`__ from T. Hoffmann and D. Brown [#SPSB2022]_ using PennyLane.

Background
----------

To optimise a machine learning (ML) model, we need to determine the gradients of a loss function
with respect to all parameters in the model. While this is a simple task for classical models, where we
can leverage the auto diff tools embedded in libraries like PyTorch or Jax, this is not a trivial task for
quantum ML models which rely on `Parametrized Quantum Circuits` (PQCs). To determine the gradients of a loss
function with respect to the circuit parameters, we are bound to numerical methods like the parameter-shift rule
or a finite-differences method. However, the computational cost of those methods scales linearly with the number of
circuit parameters, making them expensive for large models.

On the other hand, a well-established method for a black-box optimisation method for noisy and expensive objective
functions exists, namely the Simultaneous Perturbation Stochastic Approximation (SPSA) algorithm [#SPSA1987]_. SPSA
perturbs all parameters simultaneously in order to calculate an estimator of the gradient used for optimisation.

While SPSA can be used to optimize the objective of a purely quantum ML model (see `here
<https://pennylane.ai/qml/demos/tutorial_spsa.html>`_), it is rather unsuitable
for hybrid models as we lose the benefit of having access to the analytic gradients of the classical parts in the
computational graph.

Hence, the goal of the following tutorial is to walk through the process of applying the multi-variate SPSA algorithm
to PQCs to estimate Jacobians which can be hooked to the auto-diff graph of PennyLane's ML backends (PyTorch, Jax etc.).

A short intro to Automatic Differentiation (AD)
-----------------------------------------------

(Following [#SPSB2022]_ closely). Consider a simple chained function :math:`y = f(u(x))` where :math:`x\in \mathbb{R}^p`,
:math:`u: \mathbb{R}^p\rightarrow\mathbb{R}^m` and :math:`f: \mathbb{R}^m\rightarrow\mathbb{R}^n`. 

.. figure:: ../demonstrations/constant_scaling_quantum_gradient_estimation/backprop.png
   :align: center
   :scale: 30%
   :alt: backprop figure

We are interested in computing the derivative :math:`\frac{\partial y}{\partial x}` using the AD backwards pass. We start by initialising an
*upstream* vector of size :math:`\mathbb{R}^n` filled with ones. We now continue by calculating :math:`\partial y / \partial u`,
which is given by:

.. math::

    \frac{\partial y}{\partial u} = \frac{\partial y}{\partial y}\frac{\partial y}{\partial u} = \frac{\partial y}{\partial y}\frac{\partial f}{\partial u}= \mathbf{\text{upstream}}\cdot\frac{\partial f}{\partial u},

where :math:`\partial f / \partial u` is the Jacobian :math:`\mathbf{J}_f` with respect to :math:`u` (i.e. a matrix of
size :math:`\mathbb{R}^{n\times m}`). Treating the outcome as our new upstream vector
(:math:`\mathbf{\text{upstream}}\leftarrow \partial y / \partial u`), we calculate :math:`\partial y / \partial x` by:

.. math::

    \frac{\partial y}{\partial x} = \frac{\partial y}{\partial u}\frac{\partial u}{\partial x} = \mathbf{\text{upstream}}\cdot\frac{\partial u}{\partial x},

where :math:`\partial u / \partial x` is the Jacobian :math:`\mathbf{J}_u` with respect to :math:`x`
(i.e. a matrix of size :math:`\mathbb{R}^{m\times p}`).

We can repeat this procedure for an arbitrary number of inner functions and, most importantly, decompose any differentiable
function :math:`y=f(x)` into a chain of elementary operations whose derivatives we know exactly. However, while reverse-mode AD
is suitable for *classical computations*, it is not possible to backpropagate through *quantum circuits* in the same way as we don't
have access to the upstream variable due to the unobservable nature of the intermediate quantum states. We therefore need to
estimate the Jacobians of the quantum parts of our model using numerical methods. This is where SPSB comes into play.

From SPSA to SPSB
-----------------

If you are not familiar with the SPSA algorithm, we advise you to read `this section <https://pennylane.ai/qml/demos/tutorial_spsa.html#simultaneous-perturbation-stochastic-approximation-spsa>`_ first.

SPSB denotes `Simultaneous Perturbation for Stochastic Backpropagation`, which applies the gradient estimation method of the
multi-variate SPSA algorithm [#SPALL1992]_ to estimate the Jacobians of PQCs and hooks them to the auto-diff graph of the PennyLane backend.

Hence, we need to approximate the Jacobian $J$ of a (potentially noisy) multi-variate function :math:`\bm{f}: \mathbb{R}^n \rightarrow \mathbb{R}^m` where the set :math:`\mathbf{x}=\{x_1, \dots, x_n\}` denotes the set of all parameters:

.. math::

    J =
    \begin{bmatrix}
    \frac{\partial \bm{f}}{\partial x_1} & \cdots & \frac{\partial \bm{f}}{\partial x_n}
    \end{bmatrix}
    =
    \begin{bmatrix}
    \nabla^T f_1 \\
    \vdots\\
    \nabla^T f_m
    \end{bmatrix}
    =
    \begin{bmatrix}
    \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
    \vdots & \ddots & \vdots \\
    \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
    \end{bmatrix}.

Given a perturbation vector :math:`\mathbf{\Delta}`, sampled from a Rademacher distribution, we record
:math:`f_+ =  f(\mathbf{x}+\epsilon \mathbf{\Delta})` and :math:`f_- =  f(\mathbf{x}-\epsilon \mathbf{\Delta})`
where :math:`\epsilon` is a small shifting constant (e.g. 0.01). We can now estimate the Jacobian using

.. math::

     J \approx \frac{1}{2\epsilon} \cdot (\bm{f}_{+} - \bm{f}_{-}) \otimes \mathbf{\Delta}^{\odot -1},

where we denote by :math:`\otimes` the outer product, and by :math:`\mathbf{\Delta}^{\odot -1}` the element-wise inverse of :math:`\mathbf{\Delta}`.

As we can see, the overall computational overhead of SPSB is two extra executions of :math:`f`. Even if we decide to
use some `smoothing`, i.e. to sample :math:`J` multiple times with different :math:`\mathbf{\Delta}` and take the average,
the overhead is still independent of the number of function parameters :math:`n`.

Optimisation of QCNN model
--------------------------
"""
import torch
from torch import nn

import numpy as np
import pennylane as qml
from torchvision import datasets, transforms
from tqdm import trange

##############################################################################
# Define hyperparameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
N_SAMPLES = 1000
N_QUBITS = 4
N_LAYERS = 1
BATCH_SIZE = 50
N_EPOCHS = 2
LEARNING_RATE = 0.01
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


##############################################################################
# Fetch and preprocess MNIST data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((4, 4), antialias=None),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)
mnist = datasets.MNIST("./data", train=True, download=True, transform=transform)

# Get only 6s and 3s
idx = np.where((mnist.targets == 6) | (mnist.targets == 3))[0]
mnist.data = mnist.data[idx]
mnist.targets = mnist.targets[idx]

# Limit to N_SAMPLES
mnist.data = mnist.data[:N_SAMPLES]
mnist.targets = mnist.targets[:N_SAMPLES]

# Change labels to 0 and 1
mnist.targets[mnist.targets == 6] = 0
mnist.targets[mnist.targets == 3] = 1

##############################################################################
# Define circuit
# ^^^^^^^^^^

dev = qml.device("default.qubit", wires=N_QUBITS)


def get_circuit(differentiator):
    @qml.qnode(dev, diff_method=differentiator, interface="torch")
    def circuit(inputs, weights):

        # Encoding
        for i in range(N_QUBITS):
            qml.RY(inputs[i], wires=i)

        # Classifier
        for l in range(N_LAYERS):
            layer_params = weights[N_QUBITS * l : N_QUBITS * (l + 1)]
            for i in range(N_QUBITS):
                qml.CRZ(layer_params[i], wires=[i, (i + 1) % N_QUBITS])
            for i in range(N_QUBITS):
                qml.Hadamard(wires=i)

        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

    return circuit


##############################################################################
# Define torch model
# ^^^^^^^^^^


class Model(nn.Module):
    def __init__(self, differentiator, n_wires, n_layers) -> None:
        super().__init__()
        weight_shapes = {"weights": (n_layers * n_wires,)}

        circuit = get_circuit(differentiator)
        self.quanv = qml.qnn.TorchLayer(circuit, weight_shapes)

        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear(N_QUBITS * 2 * 2, 2)
        self.softmax = nn.Softmax(dim=1)
        self.final = nn.Sequential(self.flatten, self.linear, self.softmax)

    def forward(self, x):

        bs = x.shape[0]
        img_size = x.shape[1]

        all_circuit_outs = []

        for j in range(0, img_size, 2):
            for k in range(0, img_size, 2):
                image_portion = torch.cat(
                    (
                        x[:, j, k],
                        x[:, j, k + 1],
                        x[:, j + 1, k],
                        x[:, j + 1, k + 1],
                    )
                )

                data = torch.transpose(image_portion.view(4, bs), 0, 1)

                circuit_outs = self.quanv(data)
                all_circuit_outs.append(circuit_outs)

        x = torch.cat(all_circuit_outs, dim=1).float()
        x = self.final(x)
        return x


##############################################################################
# Define training loop
# ^^^^^^^^^^


def get_losses(differentiator):

    trainloader = torch.utils.data.DataLoader(
        mnist, batch_size=BATCH_SIZE, shuffle=True
    )

    model = Model(differentiator, N_QUBITS, N_LAYERS)
    model.to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss = nn.CrossEntropyLoss()

    # Initial evaluation
    initial_loss = 0.0
    initial_acc = 0.0

    for batch_idx, (data, target) in enumerate(trainloader):
        data = data.squeeze().to(DEVICE)
        target = target.to(DEVICE)
        output = model(data)
        l = loss(output, target)
        initial_loss += l.item()
        initial_acc += output.argmax(dim=1).eq(target).sum().item()

    initial_loss /= len(trainloader)
    initial_acc /= len(trainloader)

    losses = [initial_loss]
    avg_accs = [initial_acc]

    # Circuit evaluations
    init_circuit_evals = dev.num_executions
    circuit_evals = [0]
    circuit_evals_accs = [0]

    # Training loop
    for epoch in trange(N_EPOCHS, desc="Epochs"):
        running_acc = 0
        for batch_idx, (data, target) in enumerate(trainloader):
            data = data.squeeze().to(DEVICE)
            target = target.to(DEVICE)
            output = model(data)

            l = loss(output, target)
            losses.append(l.item())

            opt.zero_grad()
            l.backward()
            opt.step()

            accuracy = output.argmax(dim=1).eq(target).sum().item()
            running_acc += accuracy * len(target)
            circuit_evals.append(dev.num_executions - init_circuit_evals)

        avg_accs.append(running_acc / len(trainloader.dataset))
        circuit_evals_accs.append(dev.num_executions - init_circuit_evals)

    return losses, avg_accs, circuit_evals, circuit_evals_accs


##############################################################################
# Run training loop and plot data
# --------------------------

differentiators = ["spsa", "parameter-shift"]
losses = []
accs = []
circuit_evals = []
circuit_evals_accs = []

for differentiator in differentiators:
    loss, acc, circuit_eval, circuit_eval_accs = get_losses(differentiator)
    losses.append(loss)
    accs.append(acc)
    circuit_evals.append(circuit_eval)
    circuit_evals_accs.append(circuit_eval_accs)


##############################################################################
# Define rolling average for plotting
# ^^^^^^^^^^

def rolling_avg(data, window_size):
    convolution = np.convolve(data, np.ones(window_size,)/window_size, "valid")
    return np.append(data[: window_size - 1], convolution)

##############################################################################
# Plot losses and accuracies
# ^^^^^^^^^^
from matplotlib import pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
ax2 = ax.twinx()

window_sizes = {"spsa": 10, "parameter-shift": 3}

for i, d in enumerate(differentiators):
    rolling_loss = rolling_avg(losses[i], window_sizes[d])

    ax.plot(circuit_evals[i], rolling_loss, label=f"{d} loss")
    ax2.plot(circuit_evals_accs[i], accs[i], label=f"{d} accuracy")

ax.set_xlabel("Circuit Evaluations")
ax.set_ylabel("Loss")
ax2.set_ylabel("Accuracy")
ax.legend()

######################################################################
# References
# ----------
#
# .. [#SPSA1987] Spall, J. C. (1987).
#    *A Stochastic Approximation Technique for Generating Maximum Likelihood Parameter Estimates*.
#    Proceedings of the American Control Conference, Minneapolis, MN, June 1987, pp. 1161–1167.
#
# .. [#SPALL1992] Spall, J. C. (1992).
#    *Multivariate Stochastic Approximation Using a Simultaneous Perturbation Gradient Approximation*.
#    IEEE Transactions on Automatic Control, vol. 37(3), pp. 332–341.
#
# .. [#SPSB2022] Hoffmann, Thomas and Brown, Douglas (2022).
#    *Gradient Estimation with Constant Scaling for Hybrid Quantum Machine Learning*.
#    arXiv preprint, `2211.13981 <https://arxiv.org/abs/2211.13981>`__.
#
