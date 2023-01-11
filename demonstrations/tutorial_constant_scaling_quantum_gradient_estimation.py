r"""
Gradient Estimation with Constant Scaling for Hybrid Quantum Machine Learning
=============================================================================

.. meta::
    :property=“og:description”: Implement a quantum hybrid model optimized using SPSA gradients
    :property=“og:image”: https://pennylane.ai/qml/_images/spsa_mntn.png

*Author: Thomas Hoffmann — Posted: 11 January 2023. Last updated: 11 January 2023.*

In this tutorial, we show how to use the gradient method from SPSA to
optimize hybrid quantum-classical neural networks. Research results are
published in
`Gradient Estimation with Constant Scaling for Hybrid Quantum Machine Learning
<https://arxiv.org/abs/2211.13981>`__
from T. Hoffmann and D. Brown [#SPSB2022]_ using PennyLane.

"""


######################################################################
# References
# ----------
#
# .. [#SPSB2022] Hoffmann, Thomas, Brown, Douglas (2022).
#    *Gradient Estimation with Constant Scaling for Hybrid Quantum Machine Learning*.
#    arXiv preprint, `2211.13981 <https://arxiv.org/abs/2211.13981>`__.
#