# Lemmas and Ideas

## Useful Laws, Formulas, ect...

### Random walks

In a random wlk on the number line, say $(0,N)$ then the probability of getting to $N$ starting from $0<x<N$ is $x/N$. This generalizes lineraly over hte number line. (Proof is through creating a functional equation and then proving it's the unique solution).

The expected number of steps to quit this range is $x * (N-x)$

### Linearity of Expectation

If you have N Indep variables then : 
$E(X_1 + ... + X_N) = \sum E(X_i)$  

### Bayes

$$P(A|B) = \frac{P(A)*P(B|A)}{P(B)}$$

### Kelly Criterion

If you are in a game with $b : 1$ odds, you should bet $f = \frac{bp-q}{b}$ of your fortune every game. $p$ being your winning % and $q$ your loosing percentage.

## Ideas

### Distribution

**idea 1:** suppose variables X_i follow an unspeciffied continous distribution, the probability of one being max of them all is $\frac{1}{i}$ since they all have the same distribution. (And so, if we add an $i+1$-th variable, the chance it's the max is $\frac{1}{i+1}$ instead of say $\frac{1}{2^i}$ for example).

**idea 2 :** if you have $X$ type-1 objects and $Y$ type-2 objects, you can use type-1 objects as separators to get the average number of type-2 objects between each roll of a type-1 object : $\frac{Y}{X+1}$

**idea 3 :** The following two distribs are not the same : Randomly putting X 1's and Y 0's in a sequence vs putting baskets inbetween (plus at start and end) of 0's and then randomly placing 1's inside those. \
This is due to the was groupings work. If you have two 1's (say one is red and one is blue) and two 0s (one red one blue), in the first distrib, you have 4 ways of getting 1100. (RBRB,BRRB ect). P(1100) = 4*/4!=1/6 = 2!2!1/4! \
In the 2nd distrib however, you only have 1 way of getting 1100. P(1100) = (1/3)^2.

### Circular permutations

**Idea 1:** Suppose you have X red beans and Y blue beans to put in a braclet. To find the number of unique bracelets you can think of **gaps between red/blue beans** modulo the rotation (2 2 1) != (2 1 2) but (2 1) == (1 2).

### Dice rolls

**idea 1:** Getting a specific score in dice rolls. Suppose you wanna get a number that is > max_score(1Dice). then you can use the *stones & sticks* methode on the difference btw the max score and the dices thrown to get the number of possible combinations.

It can also be generalized to include numbers < max_score(1Dice)

**idea 2** Expected value of multiple rolls : You can start with expected value at the last roll and then determine what would hten be the expected value of the one before it ect...

If looking for a number to chose as the minimal number before rolling, you can try to maximze a function.

### Problem specific ideas : 

**idea 1** Suppose i randomly come in a 1 hour range and stay for X mins. and another person comes in the same range and stays for Y mins. The proba of us meeting can be found by graphing a square with Y=X being that we meet at that time.
Then for each y=X I can either have come at most X mins before or that he left at most Y mins after. this gives us a surface of two triangles with a ruban between them. that ruban's area (relative to that of the square) presents the proba of us meeting. $(60*60-X^2/2-Y^2/2)$

