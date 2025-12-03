# ESL Problems

This file contains my thoughts/commentairy on attempted problems in ESL, calculations are only included when there's added value. I might occasionally refer to [this solution manual](https://waxworksmath.com/Authors/G_M/Hastie/WriteUp/Weatherwax_Epstein_Hastie_Solution_Manual.pdf).

## Chapter 2:

**P1:** Trivial problem. The condition of $ \hat{y} $ summing to 1 isn't really needed... You basically just minimize the distance formula $\sqrt{\sum_{i \not = k} x_i^2 + (x_k -1)^2}$ , this leads to maximizing $x_k$\

**P2:** The basic idea is simple, you basically graph $P(C_1 | X) = P(C_2 | X)$, which ultimately is $PDF_{C_1} = PDF_{C_2}$...\
Now the author gave us 2 ways of generating data, either 2 gaussians with the same variance, one at (0,1) the other (1,0) and the second is first generating 10 pts each from 2 gaussians : with the same centers as the previous case. After that, we generate for each set of points a gaussian at the center of every point (with low varience)...\
It should be stated that we're talking about covariance matrices that are of the form $C* I_2$... \
For the first distribution, the boundary will be in the form of a line (anyone wants a geometrical proof with power of a point?? ;)) ).\ 
The second case however is more interesting, our $PDF_1$ is a sum of multiple gaussian $PDFs$ with different mean vectors... \
Beyond what the problem is asking, an interesting thing to check is how this bayes boundary changes when we change the variance in the 2 mean vectors case (lines/quadratic curves...). These are derived from equating the two $PDF$ formulas in their general form.\

Note : this only is the case when the priors are equal (in our case, say we decided to generate equal number of points from 1st class as second... If that was not the case, then our equation changes. A more general one is $\pi_1 * PDF_1 = \pi_2 * PDF_2$)

**P3:** This was also a simple one, (precising the difficulty at each exercise because ESL's problems have this reputation on the web of being hard, so I'm trying to keep notes of which ones are actually hard... ). The solution comes from calculating the cumulative distribution function : notice that the probability of a point being in the $r$-radius $p$-dimenssioned sphere is $r^p$ (sphere volume divided by unit sphere volume).\
And so, the proba of no point being in it is $(1 - r^p)^N$ which makes $P(min(d) \leq x) = 1 - (1 - r^p)^N $. Setting this CDF equal to $\frac{1}{2}$ gives us the desired result...\
Another interesting question would be calculating the mean, in which case we can either derive our CDF and integrate $\int_0^1 PDF(r) r \hspace{0.1cm} dr$, or just integrate $\int_0^1 1 - CDF(x)dx$

**P4:** This question, although scarry at first glance is pretty simple. The 2nd part i.e. (the target point has expected squared distance of &p&) directly follows from the $\Chi^2$ distribution.\
As for the first part, we are basically examining $\sum^p a_i \cdot v_i$ with $v$ being a training point.\
We have $v_i ~ N(0,1) \forall v_i$ this means our sum follows a law $N(\sum (a_i \cdot 0) , \sum (a_i^2 \cdot 1))$. Since $a$ is a unit vector, $E(a_i) = E(\frac{x_i}{|| x ||})$, we can say that the $z_i$s follow a $N(0,1)$ distribution

**P5:** Check my course notes.