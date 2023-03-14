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

.. figure:: ../demonstrations/data_reuploading/backprop.png
   :scale: 65%
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
