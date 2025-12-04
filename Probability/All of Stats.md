# All of Stats

[Book link](https://egrcc.github.io/docs/math/all-of-statistics.pdf)

The study of this book was meant as a support to the study of Elements of Statistical Learning (Check notes); this doc is updated on a consult basis...

Titles : 
- Convergence of Random Variables
- Bayesian Inference
- Statistical Decision Theory
- Linear and Logistic Regression

## Convergence of Random Variables : 
We will be discussing 2 main ideas in this part : 
- Law of large numbers : The sample average converges in **probability** to the expectation $\bar X_n = n^-1 \sum X_i$ converges to $\mu = E(X_i)$ i.e. the sample average is close to $\mu$ with high probability...
- Centra limit Theorem : $\sqrt{n} (\bar X_n - \mu)$ converges in **distribution** to a Normal distribution, i.t. the sample average has approximately a Normal distribution for large n...

We start by giving a rigurous definition of these convergences : 
- We say that $X_n$ converges in probability to $X$ if : $P(|X_n - X| > \epsilon) \left 0 $ as $ n \left \inf$