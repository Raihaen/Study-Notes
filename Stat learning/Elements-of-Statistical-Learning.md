# Elements of Statistical Learning
The following are my notes from [Elements of statistical learning](https://www.sas.upenn.edu/~fdiebold/NoHesitations/BookAdvanced.pdf).
First of all, it should be noted that I relied on Max Dama's [recommendations on most useful chapters of the book](https://blog.headlandstech.com/2022/02/16/elements-of-statistical-learning-8-10/) (I was lucky enough to have a conversation with him). 

Next, it should be noted that at the time of writing these notes, I'm still a novice in AI(which isn't hopefully the case by the time you're reading this), and so, directly approaching ESL wouldn't allow me to fully benefit from it, and this is why I started with another useful book for the same purpose (more elementary but with lesser depth) -- [Introduction to Statistical learning](https://www.stat.berkeley.edu/~rabbee/s154/ISLR_First_Printing.pdf), these notes also go through a few of its chapters, especially at the beguinning (I'll make sure to always say which is which).

To summerize, I will start with ISL and then switch to ESL once I feel like I have a sufficient mastery of the necessary topics.

## Chapter I [ESL]
Supervised learning is when we have an outcome variable to 'guide the learning process', thus we compare our model's result $\hat y$ to some desired result $y$.
Unsupervised learning on the otherside, our task is to describe how the data is organized/clustered.

## Chapter I [ISL]
Example on usecase of unsupervised learning : divide a demgraphic into groups to run ads on.

### A brief history of statistical learning:
Start of 19th century, *least squares* method was develloped, impplementing an early form of *linear regression (LR)*. What followed this was *linear discriminant analysis* (an application of LR for qualitative values). After that, *logistical regression* was develloped, then a *general linear model* was developed, an entire class of SL methods with LR and logReg as special cases.\
More methods were developped in the 70s, but were still linear. It wasn't untill the 80s that we had enough computation power for *non-linear methods* : *classification and regression trees* were develloped, followed by *generalized additive models*. *Neural networks* got some traction in the 80s and *support vector machines* in the 90s.\
Nowadays, this set of techniques became a separate filed focused on supervised and unsupervised modeling and predictions : statistical learning

### Notation (updated whenever needed):
Let $n$ be the number of observations, $p$ the number of variables. Let $\text{X}$ the matrix representing observations of variables, sucht that $x_{ij}$ is the $i$-th observation's $j$-th variable value.
Remeber that vectors are always repsented as columns. and so, $\text{x}_j$ would be the set of observations for the $j$-th variable. We can represent $\text{X}$ as a vector of observation columns : $\text{X} =$ { $x_i^T$ }. 

## Chapter 2 [ISL]
Inputs $X_1, ..., X_p$ go by different names : *predictors, features, independent variables...*, the output of the model is called the *dependent variable* or *response*.\
Let $X=(X_1,X_2,...,X_p)$ be the set of feature variables.\
To make a model, we assume there is a relationship between features and the response : $Y=f(X)+ \epsilon$ for a fixed but unknown function $f$ and a **random error term** $\epsilon$, independent of $X$ and with mean 0.\
Think of $f$ as the **systematic** information $X$ gives on $Y$.

Why do we estimate $f$? This is due to 2 reasons : prediction and inference.
### Prediction :
- Reductible vs Irreductible error : Our prediction's error comes from two different sources : one can be reduced through better models (reductible) and that's by making our $\hat f$ the closest to $f$ possible. However, even if our $\hat f$ was perfect, there's still an error non dependent on $X$, which we symbolize by $\epsilon$, this error is out of our control and so is irreductible. It might come from some other predictor variables we didn't account for, or simply from chance and randomness (individual medication's minor production differences, dice rolls, etc...) [^1].\
We can write [^2] : $$E[{(Y- \hat Y)}^2] = \underbrace{{(f(X) -\hat f(X))}^2}_{\text{reductible}} + \text{Var}(\epsilon)$$




# Footnotes :
- [^1] -- free will ?.
[^2] -- proof : $E[{(Y- \hat Y)}^2] = E[(f + \epsilon - \hat f)^2] = E[f^2 + \hat f ^2 + \epsilon ^2 + 2 \epsilon (f-\hat f) - 2 (f \cdot \hat f)] = E[(f-\hat f)^2] + E(\epsilon ^2 ) + 2 E(\epsilon) \cdot E(f - \hat f)$. -- since $\epsilon$ is independent from both $f$ and $\hat f$.\
We get : $ E[(f-\hat f)^2] + E(\epsilon ^2 ) - E(\epsilon)^2 = (f-\hat f)^2 + \text{Var}(\epsilon)$ since $E(\epsilon) = 0$ 