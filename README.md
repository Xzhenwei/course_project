# PAI project
 PAI course project
There will be 4 projects here with their summaries

Reference for task 2: https://gluon.mxnet.io/chapter18_variational-methods-and-uncertainty/bayes-by-backprop.html

Task3 description:
In this project we are dealing with bayesian optimization, the optimization algorithm has been established and given by the TA. We are focusing on scripting the surrogate function and acquisition function only.

We could browsed a lot of samples from the internet about the unconstrained bayesian optimization problems, but in this project we need to deal with the constrained ones. We then incorporate our prior believes regarding the objective model and the constraint model. We utilized a mixed-kernel prior given the parameters from the project post.

Now comes to the acquisition function, we referred to the paper, Bayesian Optimization with Inequality Constraints, reference is http://proceedings.mlr.press/v32/gardner14.pdf. The extrapolations of the mathematical formulae is listed in part 3.1 Intuitively viewing on the hybrid acquisition function is, we need to contain the probability with the constraint is satisfied also in the formula, so we need to time the pdf of the constrant model also in the formula.

To get the solution, we only need to send the best solution from the previous point. The values are given from the acquisition functions. 
