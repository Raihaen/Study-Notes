# ESL Problems

This file contains my thoughts/commentairy on attempted problems in ESL, calculations are only included when there's added value. I might occasionally refer to [this solution manual](https://waxworksmath.com/Authors/G_M/Hastie/WriteUp/Weatherwax_Epstein_Hastie_Solution_Manual.pdf).

## Chapter 2:

**P1:** Trivial problem. The condition of $ \hat{y} $ summing to 1 isn't really needed... You basically just minimize the distance formula $\sqrt{\sum_{i \not = k} x_i^2 + (x_k -1)^2}$ , this leads to maximizing $x_k$\

**P2:** The basic idea is simple, you basically graph $P(C_1 | X) = P(C_2 | X)$, which ultimately is $PDF_{C_1} = PDF_{C_2}$...\
Now the author gave us 2 ways of generating data, either 2 gaussians with the same variance, one at (0,1) the other (1,0) and the second is first generating 10 pts each from 2 gaussians : with the same centers as the previous case. After that, we generate for each set of points a gaussian at the center of every point (with low varience)...\
It should be stated that we're talking about covariance matrices that are of the form $C* I_2$... \
For the first distribution, the boundary will be in the form of a line (anyone wants a geometrical proof with power of a point?? ;)) ).\ 
The second case however is more interesting, our $PDF_1$ is a sum of multiple gaussian $PDFs$ with different mean vectors... \
Beyond what the problem is asking, an interesting thing to check is how this bayes boundary changes when we change the variance in the 2 mean vectors case (lines/quadratic curves...). These are derived from equating the two $PDE$ formulas in their general form.