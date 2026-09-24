# PHYS 520 - Homework 2: The Cowles-Jones Ratio

## The big picture

The two papers and the algebra expose two different ways that an independent price process can appear persistent:

1. **Time averaging:** even if the underlying price is a random walk, replacing point observations by averages or midranges inside each time block induces positive serial correlation in the changes of the reported series.
2. **Drift:** even if successive directions are independent, unequal probabilities of rising and falling make equal consecutive signs more common than unequal signs.

Thus, observing more sequences than reversals does not by itself establish genuine temporal dependence.

## 1. Reading

### What Cowles and Jones did in 1937

Cowles and Jones asked whether stock-price changes show "inertia." A **sequence** is two consecutive changes in the same direction, while a **reversal** is two consecutive changes in opposite directions. Under their simple null model of independent, equally likely rises and falls, sequences and reversals should be equally common, so the ratio

\[
C=\frac{N_s}{N_r}
\]

should be near one.

They studied intervals ranging from 20 minutes to 10 years. Their main findings were:

- For every interval shorter than four years, sequences outnumbered reversals.
- The reported monthly result was especially striking: 748 sequences and 450 reversals in 1,200 observations, so \(C=1.66\) and the empirical probability of a sequence was about \(0.625\).
- They also found persistence in the relative performance of industry groups and individual stocks.
- A rule that followed the previous price direction looked best at a one-month horizon. With their assumptions about trading costs, they estimated an average annual net gain of roughly 7 percent, although they also estimated about a one-in-three chance of losing money in any particular year.

Their conclusion was cautious but still strong: the strategy did not promise consistent or large profits, and the favorable monthly interval had been selected after examining 26 possible intervals, but the excess of sequences was presented as evidence of structure in stock prices.

### Why Cowles published a revision in 1960

The original analysis failed to distinguish **point-in-time prices** from **averages within a time interval**. Part of the monthly rail index was constructed from the average of each month's high and low. This is not an innocent transformation.

Let the unobserved underlying price follow a random walk. If it rises through January, January's average tends to be above December's average but below February's average. Consequently both measured changes, December-to-January and January-to-February, tend to be positive. A falling January similarly tends to produce two negative measured changes. The averaging operation therefore creates sequences even though the underlying increments are independent.

The Davis simulation reported on page 911 makes the point directly. From a 1,200-step random walk, selecting every twelfth point gave 50 sequences and 48 reversals, essentially the random-null result. Using the midpoint of the high and low in each 12-point block gave 58 sequences and 40 reversals.

Cowles recomputed the statistics using a specified point, such as the close or first trading day, for each period. The old monthly count of 748 sequences and 450 reversals fell to 530 and 466. Thus the dramatic monthly evidence was largely an artifact of time averaging. Cowles regarded the old monthly conclusion as invalid, although his point-sampled daily, weekly, and annual results still showed smaller excesses of sequences. He also stressed that the remaining persistence was too small to yield meaningful profits after brokerage costs.

## 2. Numerical part

The accompanying script constructs exactly the experiment described on page 911:

1. Draw 1,200 independent increments uniformly from \([-1,1]\).
2. Cumulatively sum them to form a random walk.
3. Divide the walk into 100 blocks of 12 observations.
4. Compare three annualized series: every twelfth point, the midpoint of each block's high and low, and the mean of all 12 values.
5. Count sequences and reversals in the 99 first differences, giving 98 adjacent pairs.

Run:

```bash
python3 homework2/cowles_jones_simulation.py
```

With the fixed single-run seed, the main comparison is:

| Construction | \(N_s\) | \(N_r\) | \(C\) | lag-one correlation |
|---|---:|---:|---:|---:|
| Every twelfth point | 49 | 49 | 1.000 | -0.023 |
| Midpoint of block high and low | 59 | 39 | 1.513 | 0.215 |
| Mean of all 12 block values | 59 | 39 | 1.513 | 0.137 |

This single run is close to Davis's 50/48 versus 58/40 comparison. It is illustrative, not a statistical estimate. In 20,000 independent repetitions, the mean results are approximately:

| Construction | \(E[N_s]\) | \(E[N_r]\) | \(E[C]\) | pooled \(C=E[N_s]/E[N_r]\) | mean lag-one correlation |
|---|---:|---:|---:|---:|---:|
| Every twelfth point | 49.055 | 48.946 | 1.023 | 1.002 | -0.009 |
| Midpoint of block high and low | 57.734 | 40.266 | 1.469 | 1.434 | 0.270 |
| Mean of all 12 block values | 56.782 | 41.218 | 1.411 | 1.378 | 0.233 |

The endpoint sample behaves as expected for independent block returns. Both within-block summaries create positive first-order correlation and a large excess of sequences. Notice also that the average of the ratios, \(E[N_s/N_r]\), is not the same as the ratio of aggregate counts, \(E[N_s]/E[N_r]\).

## 3. Algebraic part

Assume \(I_t\) are independent Bernoulli variables with

\[
P(I_t=1)=p,\qquad P(I_t=0)=q=1-p.
\]

There are \(n+1\) returns and hence \(n\) adjacent pairs.

### (a) Indicator for a sequence

Define

\[
Y_t=I_tI_{t+1}+(1-I_t)(1-I_{t+1}).
\]

If both variables equal 1, the first product is 1. If both equal 0, the second product is 1. If they differ, both products vanish. Therefore

\[
Y_t=\mathbf 1\{I_t=I_{t+1}\},
\]

so \(Y_t=1\) for a sequence and \(Y_t=0\) for a reversal. Counting the sequence indicators gives

\[
\boxed{N_s=\sum_{t=1}^{n}Y_t},\qquad N_r=n-N_s.
\]

### (b) One-point moments

A sequence is either two rises or two falls. Independence gives

\[
P(Y_t=1)=P(11)+P(00)=p^2+q^2\equiv\pi,
\]

and

\[
P(Y_t=0)=P(10)+P(01)=2pq=1-\pi.
\]

Because \(Y_t\) is an indicator, \(Y_t^2=Y_t\). Hence

\[
\boxed{\langle Y_t\rangle=\pi},\qquad
\boxed{\langle Y_t^2\rangle=\pi},\qquad
\operatorname{Var}(Y_t)=\pi(1-\pi).
\]

### (c) Adjacent two-point moment

The event \(Y_tY_{t+1}=1\) requires both adjacent pairs to be sequences. Thus all three variables must agree: either 111 or 000. Therefore

\[
\boxed{\langle Y_tY_{t+1}\rangle=p^3+q^3}.
\]

Writing \(a=p^3+q^3\), the adjacent covariance is

\[
\operatorname{Cov}(Y_t,Y_{t+1})=a-\pi^2.
\]

For \(|t-s|>1\), the two indicators depend on disjoint sets of independent \(I\)'s and are therefore independent.

### (d) Large-\(n\) distribution of \(N_s\)

The variables \(Y_t\) are bounded and **1-dependent**: only neighbors can be correlated. A central limit theorem for such short-range-dependent variables implies that their sum is asymptotically Gaussian.

The exact mean is

\[
\boxed{\langle N_s\rangle=n\pi}.
\]

Its variance contains \(n\) individual variances and \(n-1\) adjacent covariance pairs, each appearing twice in the double sum:

\[
\boxed{
\sigma^2=\operatorname{Var}(N_s)
=n\pi(1-\pi)+2(n-1)\left[p^3+q^3-\pi^2\right].
}
\]

For large \(n\),

\[
\sigma^2\simeq n\left\{\pi(1-\pi)+2[p^3+q^3-\pi^2]\right\}.
\]

Thus

\[
N_s\ \dot\sim\ \mathcal N(n\pi,\sigma^2).
\]

### (e) Distribution of the Cowles-Jones ratio

Let \(X=N_s/n\). Then \(C=g(X)\), where

\[
g(x)=\frac{x}{1-x},\qquad g'(x)=\frac{1}{(1-x)^2}.
\]

For large \(n\), \(X\) fluctuates only by order \(n^{-1/2}\), so a first-order Taylor expansion about \(\pi\) gives

\[
C\simeq g(\pi)+g'(\pi)(X-\pi).
\]

Consequently \(C\) is asymptotically Gaussian, with leading-order mean

\[
\boxed{\langle C\rangle\simeq\frac{\pi}{1-\pi}}
\]

and variance

\[
\begin{aligned}
\Sigma^2
&\simeq [g'(\pi)]^2\operatorname{Var}(N_s/n)\\
&=\boxed{
\frac{\pi(1-\pi)+2[p^3+q^3-\pi^2]}
{n(1-\pi)^4}
}.
\end{aligned}
\]

This is the stated result. The true finite-sample mean has a small \(O(1/n)\) correction because a nonlinear function does not generally satisfy \(E[g(X)]=g(E[X])\).

### (f) Drift and interpretation

Since

\[
\pi=p^2+q^2=1-2pq\geq\frac12,
\]

with equality only at \(p=1/2\),

\[
\boxed{
\langle C\rangle\simeq\frac{p^2+q^2}{2pq}\geq1.
}
\]

This is true for both \(p>1/2\) and \(p<1/2\): whichever outcome is more common is also more likely to appear twice consecutively.

If the signed random-walk increment is \(S_t=2I_t-1\in\{-1,+1\}\), its drift is

\[
\boxed{\mu=E[S_t]=2p-1}.
\]

In terms of drift,

\[
\pi=\frac{1+\mu^2}{2},\qquad
\boxed{\langle C\rangle\simeq\frac{1+\mu^2}{1-\mu^2}}.
\]

Thus either positive or negative drift raises \(C\), symmetrically in \(|\mu|\), even when the directions are independent. A raw test of \(C=1\) therefore confounds serial dependence with drift. In data, one should estimate \(p\), compare the observed count with the drift-adjusted null \(n\pi\), and use the variance derived above. One must also state whether counts are pooled before taking the ratio or ratios are averaged across windows: these operations give different answers, especially for short windows, because \(N_s/N_r\) is nonlinear and can become very large when \(N_r\) is small.

## Final interpretation

The 1937 statistic was insightful but its original null model was incomplete. A large \(C\) can be caused by genuine serial dependence, within-period averaging, drift, or a combination of them. A defensible analysis therefore uses point-sampled prices, accounts for unequal rise/fall probabilities, and specifies exactly how observations are pooled.
