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

## Chapter 2 [ISL]
### A brief history of statistical learning:
Start of 19th century, *least squares* method was develloped, impplementing an early form of *linear regression (LR)*. What followed this was *linear discriminant analysis* (an application of LR for qualitative values). After that, *logistical regression* was develloped, then a *general linear model* was developed, an entire class of SL methods with LR and logReg as special cases.\
More methods were developped in the 70s, but were still linear. It wasn't untill the 80s that we had enough computation power for *non-linear methods* : *classification and regression trees* were develloped, followed by *generalized additive models*. *Neural networks* got some traction in the 80s and *support vector machines* in the 90s.\
Nowadays, this set of techniques became a separate filed focused on supervised and unsupervised modeling and predictions : statistical learning

### Notation (updated whenever needed):
Let $n$ be the number of observations, $p$ the number of variables. Let $\text{X}$ the matrix representing observations of variables, sucht that $x_{ij}$ is the $i$-th observation's $j$-th variable value.
Remeber that vectors are always repsented as columns. and so, $\text{x}_j$ would be the set of observations for the $j$-th variable. We can represent $\text{X}$ as a vector of observation columns : $\text{X} =$ { $x_i^T$ }. 

Inputs $X_1, ..., X_p$ go by different names : *predictors, features, independent variables...*, the output of the model is called the *dependent variable* or *response*.\
Let $X=(X_1,X_2,...,X_p)$ be the set of feature variables.\
To make a model, we assume there is a relationship between features and the response : $Y=f(X)+ \epsilon$ for a fixed but unknown function $f$ and a **random error term** $\epsilon$, independent of $X$ and with mean 0.\
Think of $f$ as the **systematic** information $X$ gives on $Y$.

### Why do we estimate $f$? 
(it should be understood as, why do we do statistical learning, more than why estimate $f$ itself) This is due to 2 reasons : prediction and inference.
#### Prediction :
- Reductible vs Irreductible error : Our prediction's error comes from two different sources : one can be reduced through better models (reductible) and that's by making our $\hat f$ the closest to $f$ possible. However, even if our $\hat f$ was perfect, there's still an error non dependent on $X$, which we symbolize by $\epsilon$, this error is out of our control and so is irreductible. It might come from some other predictor variables we didn't account for, or simply from chance and randomness (individual medication's minor production differences, dice rolls, etc...) [^1].\
We can write [^2] :\
$ E[{(Y- \hat Y)}^2] = (f(X) -\hat f(X))^2+ \text{Var}(\epsilon)$ -- The first part is reductible while the second one isn't.

#### Inference :
Sometimes, our goal is not to predict a phenomenon, but to study the relationships between some variables and our target. Which predictors are associated with the response ? which variables significantly affect the response? in what way ? Is this relationship linear or an other type? 

To give an example on the difference btween the two, consider a company that wants to know their product target, based on some features, they want to know who will have a buy response : this is prediction. However, on another setting, a company might be intrested in studying which media are associated with sales, and how do investements in such media convert into sales (ROI / conversion speed ect). You might also be intrested in both reasons (inference and prediction).\
One thing to note is that, complex models might allow for better predictions, but that might be at the expense of interpretabitly (and so compremising inference).

### How do we estimate $f$ ?

#### Parametric vs non parametric methods :
**Parametric** methods are basically transforming the question of estimating $f$ into estimating parameters -- say, $\beta_i$'s for $\hat f = \sum \beta_i X_i$. A downside to this is that it makes assumptions on the nature of the relations between $X_i$ and $Y$, which might make our model imprecise. A common approach is to estimate parameters by minimizing the least square error, but more metrics exist.\
On the other side, **non parametric** methods try to directly fit the response's data (quoting the book : **create an estimate to $f$ as close as possible to the observed data, subject to the fit being smooth**). And while this has the advantage of making no assumptions on the data, we run into two problems : the risk of overfitting (we usually play with the curve's smootheness to reduce it), and we need large amounts of data to create a good fitting $\hat f$. Which might not always be possible.

#### More on supervised vs unsupervised :
We can unsupervised learning by trying to hypothyse the relations between variables / observations. A usual method of treating unsupervised learning is through clustering.\
Note that **clustering $\not =$ classification**. Clustering is unsupervised, while classification is supervised (you try to assign an element to one of already defined classes).\
Clustering, although possible to be visualised with 2/3 featurs, it becomes increasingly complicated as $p$ gets higher and higher, *d'ou* the necessity of methods to handle this.\
Another intresting question, is when you have $n$ observations of the features, but only $m<n$ observations of the response. This is the case if it's expensive to get observations on $Y$. This is somewhat in the middle of both fields : a semi-supervised learning problem where you try to incorporate info from both fields.

#### Regression vs Classification
Data is either quantitative or qualitative (categorical), the type of predictor variables doesn't matter much and most methods can be used as long as the these features are properly coded.\
However, different methods are used depending on the type of dependent variable $Y$, least squares linear regression is used for quantitative $Y$s (Regression), wheras logistic regression is usually used for binary classification (a Classification method despite its name).\
Some other SL methods (such as K-nearest neighbors or boosting) can be used for either...

### Assessing model accuracy
There is no perfect model, the best models differ depending on situations...
#### Fitting the target
We usually asses model accuracy using $\text{MSE}=\frac{1}{n} \sum (y_i-\hat f)^2$ (mean squared error).\
Overfitting reffers specifically to the situation where a model with less degrees of freedom would've yielded better results. We always expect the testing accuracy to be less than the training accuracy since all models directly or indirectly try to minimize the training error.\
The book presented an excellent example on training accuracy doesn't always translate to testing accuracy, by creating some data with a curve $f$ and some noise $\epsilon$ and showing how different models with increasing degrees of freedom created a U-shaped curve in test accuracy (although the training accuracy kept decreasing)[^3]. Like already mentionned, overfitting only reffers to models with higher DF than the model with minimal test accuracy.\
In real world applications, it's usually hard to estimate the test MSE (because there's usually no test data available), and so various methods were developped to try and estimate the minimum test accuracy point. One such method is Cross validation (this will be discussed later on).

#### The Bias-Variance trade off
Suppose we train a model on multiple datasets, creating multiple $\hat f$'s. We then apply each on some $x_0$. We have :\
$\text{E}[(y_0 - \hat f(x_0))^2] = \text{Var}(\hat f(x_0)) + [\text{Bias}(\hat f(x_0))]^2+ \text{Var}(\epsilon)$
[^4].\
Notice that this sets a theoretical inf bound :  $\text{Var}(\epsilon)$. Furthermore, both square of Bias and Var of our estimator of $\hat f$ are positive, this means that minimizing them means getting them as close to 0 as possible.\
This is however complicated, as they behave differently one another :
- **Var** : As the model's felxibility increases, it starts fitting for our training set like a bespoke suit, capturing inexistant patters, and so, changing a point by another will make this "bespoke" model change drastically, thus making Var an function that increases with flexibility (/ degrees of freedom).
- **Bias** : On the other side, Bias is the difference between $f$ and our average estimation of it, and so, the more flexible the model, the closer it will fit $f$ -- except at training points, but since we're talking about the average, it will [^5] be will get closer to $f$...

The clash between these two functions gives us our trade off and the U-shaped MSE graph : eventually the rate of increase of Var will surpass that of decrease of Bias, making the MSE go up again.\
Since $f$ is inobserved, we can't really comupute test bias or var, but it's important to keep this trade-off in mind.

#### The classification setting :
Same as with regression, we can imagine a probability function that gives us the probality of having element with observations $x_0$ be in class $C$.\
We can prove that the best classifier, called the Bayes classifier, is basically choosing for each element class $C_i$ such that $\text{max}_i P(y_0 = C_i | X=  x_0 )$.\
A usual way of assessing the precision of a classification model is through the *error rate* : $\frac{1}{n} \sum I(y_i \not = \hat y_i)$. Where $I(a \not = b)$ is the indicator variable, its value is 1 if $a \not = b$, and 0 otherwise.\
The Bayes error rate is theoretical minimum, equall to $1- \text{E}(\text{max}_j P(Y=j | X))$
#### $K$-closest neighbors :
This is a simple but very useful classifier, we assign to observation $x_0$ probability of being in class $j$ based on its closest $K$ neighbors's classes : $P(Y=j | X = x_0)=\frac{\sum_{i \in N_0} I(y_i = j)}{K}$ [^6].\
Despite its simplicity, KNN can produce classifiers very close to the optimal Bayes classifier.\
We then explored how the bias-variance tradeoff manifests itself in classification (here flexibility is basically $\frac{1}{K}$)

## Chapter 2 [ESL]
Note that we will be using vector notation, so unless specified otherwise, assume vectors & matrices **(this expression needs reowking)**.
### Two simple approaches to Prediction : Least squares and KNN
#### RSS
We start by deriving the optimal $\cap \beta$ formula. We have for $\text{RSS}$ (which is basically $n \cdot \text{MSE}$) : \
$\text{RSS} = (y - X \beta)^T (y - X \beta)$, by deriving this with respect to $\beta$ :\
$(-X d\beta)^T (y - X \beta) + (y - X \beta)^T (-X d\beta)$, since $a^T b = b^T a$ (for scalar $a^T b$) then we get :\
$-2X^T((y - X \beta))$.\
To calculate $\hat \beta$ that minimizes this quantity, we set it equal to zero : $X^T((y - X \beta)) = 0$, if $X^T X$ is *nonsingular* (has an inverese), we can do :\
$(X^TX)^{-1} (X^T y - X^T X \beta) = 0$ Thus $(X^TX)^{-1} X^T y = \hat \beta$ [^7]
#### binary classification using linear regression
This presents an idea on binary classification. Imagine 2 scenarios : in the first we generate data for each class through a Gaussian distribution; in the second, we generate for each class first a set of 10 values to serve as means for Gaussians with low var.\
Interestingly, a linear decision boundary (resulting from applying linear regression) is the best we can do for scenario one [^8]. However, for the story is different for the second scenario, an optimal boundary would be nonlinear and disjoint.\
#### KNN
Continuing with scenario 2, let's explore KNN. A counter-intuitive fact is that although the number of parameters of KNN is K but the *effective* number of parameters is N/k.\
To get an idea of why, not that if the neighborhoods were non-overlaping, we will be having N/k neighboorhoods (this is because each point will be part of K differenent [cause nonoverlaping] neighbourhoods, so to not double count, we need to divide the number of points by k to get the number of neighborhoods).

#### From LR to KNN
Most of the modern models use a variation of these two Here are a few ways of how that can be done : 
- Kernel methods that give neighbours weights in smoothly descending order  (instead of the (0,1) weights of KNN).
- Distance Kernels that give some features higher weights than the others (being closer on a dimension is better than on another).
- Local regression fits the linear model to minimize a local version (locally weighted version) of RSS (instead of the constant one).
- Some models use basis expansions of features to approximate non linear functions.
- some models like *projection pursuit* and NNs use sums of non-linearly transformed linear models.

### A bit of statistical decision theory
Here we develop a small amount of theory to provide a framework/ reasoning for model developement.
#### Quantitative response :
We start by setting a metric to assess a model's performance, one such famous metric is mean squared error. Our estimated prediction error is :
$\text{EPE}(f) = E[(Y-f)^2]$\
Suppose that we know the probability of observing $P(y,x)$ then we can rewrite our equation as : \
$\text{EPE}(f) = \int (Y-f(x))\, P(dy, dx)$\
We then condition on $X$ ($P(x,y)= P(x) \, P(y|x)$), this turns our integral into :\
$\text{EPE}(f) = \int (\int (Y-f(x) \, P(dy|dx) \, P(dx)))= E_X(E_{Y|X}[(Y-f(x))^2]$ \
Since means that to optimize $f$ it suffices to optimize it locally (for each $x$) -- since the $E_X$ means each point is treated locally.\
We write : $f(x) = \text{argmin} ._c (E[(Y-c)^2] | X=x)$
Deriving and setting to zero to minimize it gives us : \
$0 = 2 E(Y-f(x)| X=x) \Rightarrow f(x) = E(Y | X=x)$. The best estimator for $f$ is the conditional mean.\

Now since we can't get this mean (as we can only do one observation per $x$), we can use an approximation (the universal approximator) : $\hat f(x) = \text{Avg}(y_i ; i \in N_k)$, $N_k$ being the region around $x$ containing $k$ points. Note that two approximations were used here -- we approximated $E(Y)$ using an average, and $x$ using a region surrounding it.\
As $N$ grows, this quantity conveges to $E(y)$. The question waiting to be asked is, since we have a universal approximator why not just use and be done with it ? Well, for 3 reasons : we often do not have a large enough $N$, furthermore, if a structured model is available, we will get a far more stable result with the same $N$, and finally, as $p$'s dimension gets larger, (the quantity still converges but) the rate of conversion decreases a lot (for large $p$, the $N$ necessary for a good enough approx is huge).
  
Another interesting note is that by changing our EPE function, say for example to $|y-c|$, the optimal approximation would become $\hat f(x) = \text{Med}(y_i ; i \in N_x)$, the median of $y$s in that region. This approximator gives better results than the average but is less used to to **its derivations not being connected** (check the orthograph).\

Lastly, let's talk about what happens when we try to model $f$ as linear; what we are doing is basically considering $f(x) \approx x^T \beta$, applying this to our initial EPE equation and deriving (in a similar way to what we did deriving RSS at the start of the chapter)[^10] leads to : 
$ f(x) = E((XX^T))^{-1} E(X Y)$. What we are doing by applying this to the train dataset is just trying to approximate this quantity [^9].\
Notice that the difference between this derivation and the one before it (for EPE), is that we didn't use conditional probability but our knowledge of the functional relationship to pool over values of X.

In conclusion, KNN and least squares solution are both approximating conditional expectations by average, BUT, this is done in 2 different ways : least squares assumes $f$ is <u> well approximated </u> globaly by a **globaly linear function**; while on the other side KNN assumes that $f$ is well approximated by a **constant function** locally.\
The flexibility of the latter has a price like we mentionned.

Many models follow in these two's footsteps (we were given the example of additive models that divide the function into the sum of one dimensional arbitrary functions)

#### Qualitative response :
We define our loss function in the same way, and our EPE. usually L where $L(g_k, g_i)$ is the price you pay for mislabeling $g_k$ as $g_i$ -- if you just consider a constant price of 1 for all mislabelings you end up with the zero-one loss function.\
Now proceeding like with quantitative responses, we start with $\text{EPE} = E[L(G, \hat G(X))]$, and the use conditional probability to prove it suffices to optimize locally and end up with :
$G(X) = \text{argmin}_g (\sum L(G_i, g) \cdot P(G_i|X=x))$.\
Now if we use the zero-one loss function, this turns into $ G(X) = \text{argmin}_g (\sum_{G_i \not = g} P(G_i|X=x))$

or just a plain $G(X) = \text{argmax}_g (P(g|X=x))$.
Seems familiar ? It's the Bias classifier ! \ 
And again, we can see that KNN is just an approximation of this classifier (same as previously discussed).

### Local methods in higher dimensions
We now expand on a point we discussed : why not just use universal estimators like KNN ?\
Well, the answer lies in the number of feature dimensions $p$, it's usually referred to as **the curse of dimensionality**.\
This _curse_ has many layers, first notice that if data is repartitionned **uniformly** (maybe wrong word), then to cover $a$ percent of the data, one would need to cover $a^{\frac{1}{p}}$ of each feature's space, for p = 10 and a = 0.01 aka 1%, you would need to explore 63% of each.\
Further more, all sample points in higher dimensions will be close to an edge of the sample (try to calculate the probability of having a point with distance $a$ to the center); the median distance to the closest point to the origin would be $(1-{\frac{1}{2}}^{\frac{1}{N}})^{\frac{1}{p}}$;\
For a function like $f(x) = e^{-8||X||^2}$, your estimate of any point will be very biased downward (since points are on edges) and RSS will be close to 1.\
Lastly, consider that to get the same data density of a $N=100$ for p=1, you'd need $N=10^{20}$ !\
In conclusion, if we want to use the nearest neighbor methods in higher dimensions with same accuracy as in lower dimensions, one would need the size of their training set to grow exponentially.

Let's now check how a linear model scales in higher dimensions :\
We start like this [^11]: $\hat \beta = (X^T X)^{-1} XY = (X^T X)^{-1} X(X^T \beta + \epsilon) = \beta + (X^T X)^{-1} X \varepsilon$.\
This yields $\hat y - y = (X^T X)^{-1} X \varepsilon$.\
Calculating the EPE and dividing it into our usual components :
$\text{EPE} = E_{x_0} E_{T}(y_0 - x_0^T \hat \beta)$ (Here $T$ represents the set of the testing samples).\
$\text{EPE(x_0)} = \text{Var}(y_0 | x_0) + \text{Bias}(\hat y_0)^2 + \text{Var}_T(\hat y_0) \$ (our usual decomposition).\
Now we know $\hat y_0$ is unbiased, and that $\text{Var}(y_0 | x_0) = \sigma^2$ is a systematic error outside the scope of the model, we thus get :\
$\text{EPE} =  \sigma^2 +  Var_T ( x_0^T \hat \beta)$\
$\text{EPE} = \sigma^2 +  Var_T ( x_0^T (X^T X)^{-1} XY)$\
$\text{EPE} = \sigma^2 +  Var_T ( x_0^T (X^T X)^{-1} X) \cdot \sigma^2$ [^12].\
This yields : 
$\text{EPE} = \sigma^2 +  E_T x_0^T (X^T X)^{-1} x_0 \sigma^2$.\
Now if $N$ is large enough and $T$ is selected at random, we can use the following approximation : $X^T X \approx N Cov(X)$. This gives us :   
$\text{EPE} = \sigma^2 +  E_T x_0^T (N \cdot Cov(X))^{-1} x_0 \sigma^2$.\
$\text{EPE} = \sigma^2 +  E_T x_0^T (Cov(X))^{-1} x_0 \frac{\sigma^2}{N}$.\
Now since $x_0^T (Cov(X))^{-1} x_0$ we can use its trace and benefit from the trace properties :\
$E_{x_0} \text{EPE}(x_0) = \sigma^2 +  E_{x_0} \text{Trace} (x_0^T (Cov(X))^{-1} x_0) \frac{\sigma^2}{N}$.\
Since Trace is cyclic, we can do : $E_{x_0} \text{Trace} (x_0^T (Cov(X))^{-1} x_0)$ equals $E_{x_0} \text{Trace} (x_0 x_0^T (Cov(X))^{-1})$.\
We move the Trace in, getting : $E_{x_0} \text{Trace} (x_0 x_0^T (Cov(X))^{-1})$









# Footnotes :
- [^1] -- free will ?.
- [^2] -- proof : $E[{(Y- \hat Y)}^2] = E[(f + \epsilon - \hat f)^2] = E[f^2 + \hat f ^2 + \epsilon ^2 + 2 \epsilon (f-\hat f) - 2 (f \cdot \hat f)] = E[(f-\hat f)^2] + E(\epsilon ^2 ) + 2 E(\epsilon) \cdot E(f - \hat f)$. -- since $\epsilon$ is independent from both $f$ and $\hat f$.\
We get : $ E[(f-\hat f)^2] + E(\epsilon ^2 ) - E(\epsilon)^2 = (f-\hat f)^2 + \text{Var}(\epsilon)$ since $E(\epsilon) = 0$ 
- [^3] -- note that for a perfect $\hat f = f$, we will have $\text{MSE}= \text{Var}(\epsilon)$ as the theoretical lower bound on MSE; this is derived from the equation on [^2].\ Ofc this theoretical lower bound is usually unknown since we dont know the shape of $f$.
- [^4] -- proof : $\text{E}[(y_0 - \hat f(x_0))^2] = \text{E}[(f(x_0)-\hat f(x_0))^2] + \text{Var}(\epsilon)$. (same as in [^2]).\
$=\text{Var}(\epsilon) + \text{E}[f(x_0) ^2 +\hat f(x_0)^2 - 2 f(x_0) \cdot \hat f(x_0)] = \text{Var}(\epsilon) + \text{E}(\hat f(x_0) ^2) - [\text{E}(\hat f(x_0)]^2 + [\text{E}(\hat f(x_0)]^2 + \text{E}[f(x_0) ^2 - 2 f(x_0) \cdot \hat f(x_0)]  $. (separating).\
$= \text{Var}(\epsilon) +  \text{Var}(\hat f(x_0)) + [\text{E}(\hat f(x_0)]^2 + f(x_0) ^2 - 2 f(x_0) \text{E}[\hat f(x_0]] = \text{Var}(\epsilon) +  \text{Var}(\hat f(x_0))  + [f-\text{E}(\hat f(x_0))]^2 $ which is through law of Bias :\
$\text{Var}(\epsilon) +  \text{Var}(\hat f(x_0)) + [\text{Bias}(\hat f(x_0))]^2$
- [^5] generally be closer to be precise, not always !
- [^6] obv, it's $\leq 1-\frac{1}{c}$ where $c$ is the total number of classes.
- [^7] We defined this as $\hat \beta$ instead of just saying $\beta$ because we assumed $X^TX$ is non singular (to get a unique estimator).
- [^8]  Proof further down the book. I'm thinking of a geometrical proof using the radical axis (or just the perpendicular bisector of [AB], A and B being the means of the two guassian distributions).
- [^9] Learning this made me so happy, insights like these are the reason I didn't want to drop ESL for ISL.
- [^10] When deriving, we get $E(X (Y - X^T \beta)) = E(XY) - E(X X^T \beta) = E(XY) - E(X X^T) \beta$, setting this to $0$ and solving for beta yields the desired result.
- [^11] Reminder that $\hat \beta = (X^T X)^{-1} XY$ while $\beta = E(X^T X)^{-1} E(XY)$, this gives us a second proof on why the $\hat \beta$ estimator is unbiased (other than the one through the relation btween the two).
- [^12] Will continue this proof later