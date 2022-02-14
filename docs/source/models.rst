nums.models
========

Classes in the ``nums.models`` module.

.. currentmodule:: nums.models.glms

Generalized Linear Models (GLMs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The GLMs are expressed in the following notation.

.. math:: f(y) = \exp\Big(\frac{y^T\cdot\theta - b(\theta)}{\phi} + c(y, \phi)\Big)

* :math:`\phi` is the dispersion parameter.
* :math:`\theta` is the parameter of a model in canonical form.
* :math:`\b` is the cumulant generating function.


The link function is expressed as follows.

.. math:: E(Y | X) = \mu

* Define the linear predictor :math:`eta = X^T \cdot \beta`
* Define :math:`g` as the link function, so that :math:`g(\mu) = eta`


.. math:: E(Y | X) = g^{-1}(eta)

The canonical link is given by :math:`g(\mu) = (b')^{-1}(\mu) = \theta`

Note, for GLMs, the mean :math:`\mu` is some function of the model's parameter.

* Normal: :math:`\mu(\mu) = \mu`
* Bernoulli: :math:`\mu(p) = p`
* Exponential: :math:`\mu(\lambda) = \frac{1}{\lambda}`
* Poisson: :math:`\mu(\lambda) = \lambda`
* Dirichlet: :math:`\mu_i(a) = \frac{a_i}{\sum a}`

:math:`\theta` is generally a function of the model parameter:

* Normal: :math:`\theta(\mu) = \mu`
* Bernoulli: :math:`\theta(p) = \ln(\frac{p}{1-p})`
* Exponential: :math:`\theta(\lambda) = -\lambda`
* Poisson: :math:`\theta(\lambda) = \ln(\lambda)`

The canonical link maps :math:`\mu` to :math:`\theta`

* Normal:
    * :math:`\mu(\mu) = \mu`
    * :math:`\theta(\mu) = \mu`
    * :math:`b(\theta) = \frac{\theta^2}{2}`
    * :math:`g(\mu) = \mu`
* Bernoulli:
    * :math:`\mu(p) = p`
    * :math:`p(\mu) = \mu`
    * :math:`\theta(p) = \ln(\frac{p}{1-p}) = \theta(\mu) = \ln(\frac{\mu}{1 - \mu})`
    * :math:`b(\theta) = \log(1 + \exp(\theta))`
    * :math:`g(\mu) = (b')^{-1}(\mu) = ln(\frac{\mu}{1-\mu}) = ln(\frac{p}{1 - p}) = \theta(p)`


.. autoclass:: GLM
    :members:
    :undoc-members:


Limited-memory BFGS
~~~~~~~~~~~~~~~~~~
Based on Nocedal and Wright, chapters 2, 3, 6 and 7.

.. automodule:: nums.models.lbfgs
   :members:
