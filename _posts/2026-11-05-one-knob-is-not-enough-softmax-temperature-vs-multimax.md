---
layout: post
title: "One Knob Is Not Enough: How MultiMax Breaks the SoftMax Temperature Trade-off"
date: 2026-11-05 00:00:00 +0000
categories: [LLM, Attention Mechanism, Machine Learning]
redirect_from:
  - "/llm/attention mechanism/machine learning/2026/11/05/one-knob-is-not-enough-softmax-temperature-vs-multimax.html"
---

At ICML 2024 we proposed **MultiMax** [1], a family of piece-wise-temperature functions that
contains SoftMax and is designed to break its trade-off between sparsity and multi-modality in
attention and output distributions. The paper makes this case mathematically, with formal
definitions of sparsity and multi-modality and a proof that a temperature trades one against the
other; the argument is rigorous but hard to follow, and I believe this is a main reason MultiMax has
been largely overlooked. This post explains it through figures and direct measurements instead of
proofs, and adds what the paper had not tested: MultiMax at the scale of LLM pretraining.

A unified SoftMax temperature can only trade off the sparsity and multi-modality of the generated
distribution: lowering it suppresses the long tail of implausible tokens but also concentrates the
mass on the top token, and raising it does the reverse (Figure 1). MultiMax replaces the unified
temperature with a piece-wise one, so that the temperature can differ from one range of logits to
another; with temperature 1 everywhere, it is SoftMax. This breaks the trade-off and pushes out the
Pareto frontier. With a low temperature below one logit threshold, where the implausible tail lies,
and a high temperature above another, where the plausible tokens lie, MultiMax suppresses the long
tail of implausible tokens while preserving the inherent diversity of plausible next tokens
(Figure 1, right).

<figure style="display: flex; flex-direction: column; align-items: center; text-align: center;">
  <img src="{{ '/assets/images/mm_vocab_temperature.png' | relative_url }}"
       alt="A toy next-token distribution at SoftMax temperatures 1, 0.7 and 0.5 and with MultiMax: lowering the temperature removes the implausible mass only by concentrating it on the top token, MultiMax removes it and keeps the plausible tokens close together"
       style="width:100%; max-width:1100px; height:auto;">
  <figcaption style="max-width: 820px; margin: 0.6em auto; text-align: left;">
    <b>Figure 1:</b> A toy next-token distribution after "The cat sat on the": six plausible tokens
    (blue) and 200 implausible ones (grey, summed into one bar). At the default temperature, the
    implausible tokens keep 9% of the mass. Lowering the temperature removes it only by concentrating
    the mass on "mat", from 0.35 at $T = 1$ to 0.60 at $T = 0.5$, where "chair" falls to 0.01.
    MultiMax, with temperature $1/3$ below the logit $b = 1$ and $1/0.7 \approx 1.4$ above $d = 1.5$,
    removes the implausible mass as $T = 0.5$ does and keeps the plausible tokens closer together
    than even the default.
  </figcaption>
</figure>

**Why one temperature cannot do both is easiest to see with two logits.** For the input $[0, x]$,
SoftMax gives the entry $x$ the probability

$$
\phi(x) = \frac{e^{x/T}}{e^{0/T} + e^{x/T}},
$$

an S-shaped curve (a sigmoid) of the difference between the two logits, here $x - 0 = x$. The same
holds in a row of any length: how SoftMax splits the mass of two entries between them depends only
on the difference of their logits, through this one curve. A temperature can only steepen or
flatten the curve everywhere; MultiMax computes the same expression with the modulated logits
$\sigma(0)$ and $\sigma(x)$ in place of $0/T$ and $x/T$, and so bends it (Figure 2, left).

On a real row of thousands of logits, the bending is measured with the two metrics of [1], which
split the entries at a logit threshold $\epsilon$ into relevant ones, like the plausible tokens of
Figure 1, and the tail. The sparsity score $S$ approaches 1 as every tail probability goes to zero;
the multi-modality score $M$ approaches 1 when the relevant entries are about as likely as the top
one, and 0 when one entry takes all the mass. Table 1 and Figure 2 (right) measure both on one
synthetic row of 2,048 logits, and [the next section](#two-metrics-fixed-once) defines them
precisely.

<figure style="display: flex; flex-direction: column; align-items: center; text-align: center;">
  <img src="{{ '/assets/images/mm_intuition_pareto.png' | relative_url }}"
       alt="Left: the probability of the entry x for two logits [0, x], where SoftMax at three temperatures steepens or flattens one S-curve and MultiMax bends it. Right: one row of 2,048 logits in the sparsity and multi-modality plane, where the SoftMax temperatures trace a frontier and MultiMax lies beyond it"
       style="width:100%; max-width:1100px; height:auto;">
  <figcaption style="max-width: 820px; margin: 0.6em auto; text-align: left;">
    <b>Figure 2:</b> <b>Left</b>: two logits $[0, x]$, plotted as the probability of the entry $x$. A
    temperature steepens (dashed, low $T$) or flattens (dotted, high $T$) the whole S-curve. MultiMax
    (red) bends it: steep below $b$, where it pushes the tail down (sparsity), and flat above $d$,
    where it leaves the other entry more mass (multi-modality). <b>Right</b>: the row of 2,048 logits
    of Table 1 in the $(S, M)$ plane. Each dot is one SoftMax temperature, coloured by $T$. The green
    quadrant is better than SoftMax($T{=}1$) in both metrics; the temperature curve never enters it,
    and MultiMax (red star) lies inside. The black arrow compares at matched sparsity: SoftMax
    reaches the sparsity of MultiMax at $T = 0.75$, with multi-modality $0.649$ against $0.866$. The
    orange square is MultiMax with term (1) only.
  </figcaption>
</figure>

| reweighting function | multi-modality $M$ ↑ | sparsity $S$ ↑ | top-1 prob. |
|---|---|---|---|
| SoftMax, $T = 4$ | 0.998 | 0.003 | 0.003 |
| SoftMax, $T = 1$ (default) | 0.792 | 0.376 | 0.213 |
| SoftMax, $T = 0.5$ | 0.515 | 0.997 | 0.489 |
| SoftMax, $T = 0.25$ | 0.298 | 1.000 | 0.705 |
| MultiMax, term (1) only, $t_b = 2$ | 0.762 | 0.728 | 0.244 |
| **MultiMax**, $t_b = 3,\ t_d = 0.3$ | **0.866** | **0.823** | 0.141 |

*Table 1: One synthetic row of $K = 2048$ logits; MultiMax uses $b = 0$ and $d = 4.5$. Both metrics
are larger-is-better and scored with one fixed choice of $\epsilon$ and $s$. No SoftMax row improves
one metric without giving up the other; MultiMax with both terms improves on $T{=}1$ in both.*

**For lack of compute, the paper's experiments stop at 156M parameters; MultiMax now lowers the
pretraining loss at 10B and at 280B parameters.** This post adds a 10B-A1B mixture-of-experts model
pretrained with and without MultiMax over 600B tokens. In pretraining the 280B-A14B production model,
MultiMax lowers the loss by more than 0.01 at around 1% training slowdown and no extra memory. The
cost stays small because MultiMax adds no layer of its own: it applies its piece-wise temperature
to the logits inside the kernel that already computes the cross-entropy loss, just before the
SoftMax.

The main results are:

- **One logit row** (Table 1, Figure 2): MultiMax improves on the default SoftMax in both metrics,
  which no temperature does: sparsity rises from $0.376$ to $0.823$ and multi-modality from $0.792$
  to $0.866$, and at matched sparsity MultiMax adds $0.217$ multi-modality.
- **GPT-2 on OpenWebText** (Figures 6 and 7): after smoothing, the validation loss is $0.008$ lower
  with MultiMax in attention and $0.019$ lower with MultiMax in attention and the output layer, on
  average from step 10k to the end of the 600k-step run.
- **A 10B-A1B mixture-of-experts model over 600B tokens** (Figures 9 and 10): the training loss,
  including a multi-token-prediction term, is $0.011$ lower in a run that differs from the baseline
  only in the vocabulary head and $0.014$ lower in a second run.

**Contents**

* TOC
{:toc}

## Two metrics, fixed once

This section defines the two metrics and fixes how they are scored, since every comparison below
depends on that choice.

Let $\phi$ map logits $x \in \mathbb{R}^K$ onto the simplex, and let $\epsilon$ be a threshold
above which an entry counts as *relevant*.

**Multi-modality** (Definition 3.2 in [1]) averages, over the $N$ relevant entries with
$\epsilon < x_n < x_{\max}$, how far each falls below the peak probability:

$$
M(x) = 1 - \frac{1}{N} \sum_{\epsilon < x_n < x_{\max}} \big( \phi(x)_{\max} - \phi(x)_n \big)
\tag{1}
$$

$M \to 1$ when the relevant entries are about equally strong, so that several modes survive, and
$M \to 0$ when one entry takes all the mass.

**Sparsity** (Definition 3.3 in [1]) averages over the $L$ tail entries with $x_l < \epsilon$:

$$
S(x) = \frac{1}{L} \sum_{x_l < \epsilon} \exp\Big( \frac{s - \phi(x)_l}{s} - 1 \Big)
\tag{2}
$$

where $s$ is a reference probability that sets the scale. Since $\frac{s - p}{s} - 1 = -\frac{p}{s}$,
each term is simply $\exp(-p_l / s)$, a smooth step that saturates at 1 once $p_l \ll s$.

**The reference $s$ must be fixed, not recomputed from the distribution being scored.** Recomputed
as $\phi_{\min}$ of each distribution, it makes $p_l / s \geq 1$ for every tail entry and caps every
term at $e^{-1}$: the $T{=}1$ SoftMax then scores $S = 9.2 \times 10^{-4}$, and the much sharper
$T{=}0.5$ even less, $3.5 \times 10^{-4}$. Throughout this post, $\epsilon$ and $s$ are computed once
from the $T{=}1$ SoftMax baseline, with $s$ the geometric mean of its tail probabilities so that the
baseline scores mid-range, and then held fixed. The paper's suggestion, $\phi_{\min}$ of the
$T{=}1$ SoftMax held fixed, leads to the same conclusions
([see below](#a-unified-temperature-can-only-trade-off)).

## The setup: one multi-modal row of 2,048 logits

All synthetic measurements use one logit row with $K = 2048$, built to be multi-modal, since a row
with a single dominant entry has nothing to be multi-modal about: four comparable peaks at
$[6.0,\ 5.7,\ 5.4,\ 5.1]$, a band of 48 mid-range entries drawn from $\mathcal{U}(0.5, 3.0)$, and a
long tail drawn from $\mathcal{N}(-2, 1)$. With $\epsilon$ set to the logit at which the baseline
assigns $p = 1/K$, the row splits into **105 relevant** and **1,942 tail** entries. The script at the
end of the post recomputes every synthetic measurement.

## A unified temperature can only trade off

This section shows that the temperature sweep traces a frontier along which sparsity and
multi-modality move in opposite directions, and measures how far MultiMax lies beyond it.

**The temperature moves the two metrics in opposite directions over its whole range.** As $T$ falls,
$M$ decreases and $S$ increases monotonically, so no temperature improves on the default in both
(Figure 2, right). The temperature is a single degree of freedom: dividing the logits by $T$ divides
the difference between any two of them by $T$, so the differences that should shrink (among the
modes) and those that should grow (between the head and the tail) change by the same factor. A high
temperature keeps mass on the tail, and a low one suppresses the runners-up. This reproduces
Proposition 3.5 of [1] numerically, including its side condition
$\epsilon \le \lVert x \rVert_1 / K$ ($-0.08$ against $2.04$ on this row).

**At matched sparsity, MultiMax adds $0.217$ multi-modality.** Comparing MultiMax ($S = 0.823$) with
SoftMax($T{=}1$) ($S = 0.376$) mixes two sparsity levels, so the fair comparison fixes the sparsity.
The temperature that reaches $S = 0.823$ is $T = 0.75$, at $M = 0.649$; MultiMax reaches it at
$M = 0.866$, above even the $0.792$ of the default. Sharpening the default to this sparsity costs
SoftMax $0.143$ in multi-modality, whereas MultiMax gains $0.074$.

**The result does not hinge on how the metrics are scored.** On this row, $M$ stays within $0.008$
of one minus the top-1 probability (Table 1). Restricted to the top eight entries,
$M$ is $0.844$ for the default, $0.920$ for MultiMax and $0.719$ for SoftMax at matched
sparsity. Scoring sparsity against the paper's $\phi_{\min}$ of the default
still leaves no temperature dominating the default, and raises the matched-sparsity gain to $0.274$.

## Piece-wise temperature: how MultiMax breaks the trade-off

This section shows that MultiMax escapes the frontier by applying different temperatures to
different logit ranges, and that its two terms control the two metrics almost independently.

MultiMax keeps SoftMax and inserts a learnable piece-wise linear modulator $\sigma$ before it:

$$
\phi_{\text{MultiMax}}(x)_i = \frac{\exp(\sigma(x_i))}{\sum_k \exp(\sigma(x_k))}, \qquad
\sigma(x) = x + \underbrace{(1 - t_b)\,\mathrm{Max}(b - x, 0)}_{\textit{term (1)}}
          + \underbrace{(t_d - 1)\,\mathrm{Max}(x - d, 0)}_{\textit{term (2)}}
\tag{3}
$$

Equivalently, $\sigma$ consists of three linear segments with slopes $t_b$, $1$ and $t_d$:

$$
\sigma(x) =
\begin{cases}
t_b x + (1 - t_b) b & x < b \\
x & b \leq x \leq d \\
t_d x + (1 - t_d) d & x > d
\end{cases}
\tag{4}
$$

The bias terms make $\sigma$ continuous, and subgradients handle the two kinks [4]. All four
parameters are learned with the network, so MultiMax adds no hyperparameter, and $t_b = t_d = 1$
recovers SoftMax exactly: a unified temperature is one slope for the whole logit range, MultiMax one
slope per range.

<figure style="display: flex; flex-direction: column; align-items: center; text-align: center;">
  <img src="{{ '/assets/images/mm_modulator.png' | relative_url }}"
       alt="The modulator function and its effect on the sorted distribution"
       style="width:100%; max-width:1000px; height:auto;">
  <figcaption style="max-width: 780px; margin: 0.6em auto; text-align: left;">
    <b>Figure 3:</b> <b>Left</b>: a unified temperature (blue) rotates the whole line about the
    origin; MultiMax (red) is steep below $b$ and flat above $d$. <b>Right</b>: the resulting sorted
    distributions on log-log axes. SoftMax at $T{=}0.5$ (blue) suppresses the tail but also spreads
    the four peaks apart; MultiMax (red) keeps the peaks nearly level, lifts the mid-range entries
    above the $T{=}1$ baseline (black, dashed) and ends far below it in the tail.
  </figcaption>
</figure>

**Each logit range gets its own slope, so each metric gets its own lever.** Across the four modes,
MultiMax falls only from $0.141$ to $0.108$, whereas SoftMax at $T{=}0.5$ falls from $0.489$ to
$0.081$, which is what a low $M$ looks like. At rank 2,000, MultiMax lies at $7 \times 10^{-9}$
against $1 \times 10^{-5}$ for the baseline, which is what a high $S$ looks like; the mass taken from
the top peaks and the tail goes to the entries in between.

### The two terms control the two metrics

**Term (1) controls sparsity and term (2) controls multi-modality.** Term (1), with $t_b > 1$ below
$b$, pushes tail entries further down; term (2), with $t_d < 1$ above $d$, shrinks the differences
among the large logits (Figure 4).

<figure style="display: flex; flex-direction: column; align-items: center; text-align: center;">
  <img src="{{ '/assets/images/mm_dominance.png' | relative_url }}"
       alt="Dominance map over the two temperature parameters"
       style="width:100%; max-width:1100px; height:auto;">
  <figcaption style="max-width: 820px; margin: 0.6em auto; text-align: left;">
    <b>Figure 4:</b> $\Delta M$ (left) and $\Delta S$ (middle) relative to SoftMax($T{=}1$) over a
    $60 \times 60$ grid of $t_b \in [1, 4]$ and $t_d \in [0.1, 1]$, with $b = 0$ and $d = 4.5$; the
    black contour is the break-even line. <b>Right</b>: where both improve at once, <b>74.5%</b> of
    the grid. $\Delta S$ is governed almost entirely by $t_b$ (vertical bands) and $\Delta M$ mostly
    by $t_d$ (near-horizontal bands). The star is the MultiMax configuration of Table 1.
  </figcaption>
</figure>

**Improving both metrics is the generic case, not a tuned corner.** About three-quarters of the grid
improves on the default in both metrics, and with the thresholds below all four peaks
($b \in [-1, 2]$, $d \in [3.5, 5]$) the share stays between 47% and 83%. One parameter moves
sparsity and the other multi-modality, whereas a unified temperature moves both in opposite
directions.

**Term (1) alone already lies beyond the frontier, but dominating the default requires term (2).**
With $t_d = 1$, term (1) raises sparsity from $0.376$ to $0.728$ at a small cost in multi-modality
($0.792 \to 0.762$; the orange square in Figure 2, right), because the peak absorbs more of the mass
taken from the tail than the runners-up do. At matched sparsity it still gains $0.082$ in $M$ over
the frontier, but only with term (2) does MultiMax improve on the default in both metrics.

### Second order adds few parameters and FLOPs

The released version adds quadratic terms, which smooth the transitions at the kinks and add
capacity:

$$
\sigma(x) = x + \sum_{n=1}^{2} \Big[ (1 - t_{b_n})\,\mathrm{Max}(b_n - x, 0)^n
                             + (t_{d_n} - 1)\,\mathrm{Max}(x - d_n, 0)^n \Big]
\tag{5}
$$

The series stops at the second order, which was sufficient in [1]. The cost is eight parameters per
layer, 96 for a 12-layer transformer (about 0.0004% of the 22M of DeiT-small [3]), and about 15
element-wise FLOPs per attention entry, about 0.04 GFLOPs against DeiT-small's 4.6 GFLOPs
(**0.9%**).

## What the model learns

The converged coefficients that [1] reports show where in depth the mechanism is used and which
term carries it (Figure 5).

<figure style="display: flex; flex-direction: column; align-items: center; text-align: center;">
  <img src="{{ '/assets/images/mm_learned.png' | relative_url }}"
       alt="Learned SegLU coefficients per layer for DeiT-small, and the modulators they define"
       style="width:100%; max-width:800px; height:auto;">
  <figcaption style="max-width: 780px; margin: 0.6em auto; text-align: left;">
    <b>Figure 5:</b> Learned coefficients of DeiT-small on ImageNet-1K, from Table 7 of [1].
    <b>Left</b>: the coefficients per layer. <b>Right</b>: the second-order modulators that they
    define via Equation (5), at three depths.
  </figcaption>
</figure>

**The mechanism strengthens with depth, and the second-order term carries it.** In DeiT-small,
$t_{b_2}$ rises from $0.99$ at layer 1 to $3.21$ at layer 12 (correlation with depth $+0.92$) and
$t_{d_1}$ falls from $2.82$ to $0.26$ ($-0.89$): stronger tail suppression and a more compressed
head with depth. The first-order tail coefficient $t_{b_1}$ does not follow (correlation $-0.35$),
so the model routes tail suppression through the quadratic term. The first two layers learn the
opposite: at layer 1, $t_{d_1} = 2.82 > 1$ spreads the head, and the modulator rises above the
identity for large logits (Figure 5, right), a preference for smoothing that clamping the parameters
to $t_b > 1,\ t_d < 1$ would have excluded.

**The language model modulates nearly an order of magnitude less.** In Table 8 of [1], the
coefficients of the language model stay within $0.353$ of the identity, against $3.094$ for
DeiT-small, for a perplexity gain of $0.7$. If a vocabulary head behaves similarly, the learned
signal is small: weight decay towards the identity could erase it.

## GPT-2 on OpenWebText: attention and the output layer each lower the loss

This section tests the unchanged ICML 2024 formulation, in attention and in the output layer, in a
full GPT-2 pretraining run with Karpathy's nanoGPT [5], using the reference implementation [2] as
released, without the training rules introduced in the next section.

<figure style="display: flex; flex-direction: column; align-items: center; text-align: center;">
  <img src="{{ '/assets/images/mm_lm_valloss.png' | relative_url }}"
       alt="Validation loss over 600k training steps for the baseline, MultiMax in attention only and MultiMax in attention plus output"
       style="width:100%; max-width:1000px; height:auto;">
  <figcaption style="max-width: 780px; margin: 0.6em auto; text-align: left;">
    <b>Figure 6:</b> Validation loss over a full 600k-iteration nanoGPT run trained from scratch:
    GPT-2 (124M) with 12 layers, 12 heads, $d_{\text{model}} = 768$, context 1024 and vocabulary
    50,304, on OpenWebText. Perplexity on the right axis. Grey: SoftMax baseline. Blue: MultiMax in
    the attention layers only. Red: MultiMax in attention <i>and</i> the output layer. Dots mark the
    lowest single evaluation of each run; evaluations scatter around the trend by about 0.007 (one
    standard deviation), so Figure 7 smooths the curves.
  </figcaption>
</figure>

<figure style="display: flex; flex-direction: column; align-items: center; text-align: center;">
  <img src="{{ '/assets/images/mm_lm_valloss_ema.png' | relative_url }}"
       alt="The runs of Figure 6 smoothed with an EMA, and the smoothed differences from the baseline"
       style="width:100%; max-width:1000px; height:auto;">
  <figcaption style="max-width: 780px; margin: 0.6em auto; text-align: left;">
    <b>Figure 7:</b> The runs of Figure 6 smoothed with an EMA with a time constant of 20k steps.
    <b>Left</b>: every evaluation (faint) and the EMA (bold); the legend gives the smoothed loss at
    step 600k. <b>Right</b>: each MultiMax run minus the baseline after smoothing, from step 10k on.
  </figcaption>
</figure>

**Both placements lower the loss** (Figures 6 and 7). The baseline reproduces the published
reference: the nanoGPT README expects this recipe to "go down to loss of ~2.85" [5], and the
smoothed grey curve ends at $2.858$. From step 10k on, MultiMax in attention alone lies $0.008$
below it on average, and MultiMax in attention and the output layer $0.019$ below; at step 600k,
the smoothed losses are $2.858$, $2.850$ and $2.844$. No output-only run was trained, so the share
of the output layer is by difference.

**The output layer adds slightly more than attention, mostly early in training.** On top of
attention, it lowers the smoothed loss by $0.011$ on average, against $0.008$ for attention itself;
before step 200k the gap is clearer, $0.013$ against $0.007$, and from step 300k on the two shares
are within $0.001$ of each other. On DeiT-small, the two placements add $0.3$ top-1 each and $0.6$
together [1], while no attention temperature, fixed or learned, beats the default by more than
$0.1$.

## From the paper to LLM pretraining

This section describes how MultiMax runs on the vocabulary head in LLM pretraining.

<figure style="display: flex; flex-direction: column; align-items: center; text-align: center;">
  <img src="{{ '/assets/images/mm_seglu_kernel.png' | relative_url }}"
       alt="SegLU inside the chunked fused cross-entropy kernel of the shared vocabulary head, used by both the main model and the MTP module"
       style="width:100%; max-width:1000px; height:auto;">
  <figcaption style="max-width: 780px; margin: 0.6em auto; text-align: left;">
    <b>Figure 8:</b> Where MultiMax runs in LLM pretraining. The main model and the
    multi-token-prediction (MTP) module share one vocabulary head, so both pass through the same
    SegLU. The kernel walks the vocabulary chunk by chunk and modulates each block of logits in
    registers before the online log-sum-exp; the eight SegLU scalars follow their own training rules.
  </figcaption>
</figure>

**The modulator is called SegLU.** The reference implementation [2] names it `SeLU`, which collides
with the established SELU (Scaled Exponential Linear Unit); here it is **SegLU**
(segmented learnable unit), with the same math:

```python
def SegLU(x, ranges, ts):
    return (x + ts[0] * relu(ranges[0] - x)       # term (1), order 1
              + ts[1] * relu(x - ranges[1])       # term (2), order 1
              + ts[2] * relu(ranges[2] - x) ** 2  # term (1), order 2
              + ts[3] * relu(x - ranges[3]) ** 2) # term (2), order 2
```

In terms of Equation (5), `ts[0]` $= 1 - t_{b_1}$ and `ts[1]` $= t_{d_1} - 1$, likewise for the
second order, and `ranges` holds $b_1, d_1, b_2, d_2$.

**FLOPs understate the cost of a naive implementation.** In eager PyTorch, every element-wise
operation of SegLU pays a kernel launch and a round trip through memory, which costs DeiT-small about
40% in training time (Table 2); fusing the chain with `torch.jit.script` cuts this to about 10%, and
the inference overhead stays below 2% even without fusion.

| implementation (DeiT-small, ImageNet-1K) | s / iteration | training overhead |
|---|---|---|
| SoftMax baseline | 0.19 | — |
| MultiMax, eager PyTorch | 0.26 | about +40% |
| MultiMax, built-in ReLU and `torch.jit.script` | 0.21 | about +10% |

*Table 2: Training time per iteration, from Section 4.4 of [1].*

**SegLU runs inside the fused cross-entropy kernel, just before its SoftMax.** At vocabulary scale,
the full logit tensor is never stored: the chunked fused linear cross-entropy (via
[Liger Kernel](https://github.com/linkedin/Liger-Kernel)) receives
`(hidden_states, weight, bias, multimax_ranges, multimax_ts)` instead of logits, and the Triton
kernel applies SegLU to each block of logits in registers, with a closed-form backward (Figure 8).
No tensor is added, so the round trips through memory behind Table 2 disappear.

**Measured in 10B-A1B pretraining, the slowdown is around 1% and peak memory does not grow.** Figure
9 compares the MultiMax run with separate clipping of the next section against a SoftMax run with
identical training settings, which differs only in its MTP block, an MoE instead of a dense MLP; the
loss baseline of the next section uses full recomputation and another attention kernel, so it is not
comparable for speed. Over steps 48.2k to 66.8k, where both runs use the same 192 GPUs and parallel
layout, the median step rate of MultiMax is $0.3117$ against $0.3148$ steps per second, $0.98\%$
lower; the SoftMax run alone ranges from $0.29$ to $0.32$ steps per second, so this bounds the
slowdown at about 1% rather than resolving it. Peak memory is $26.6$ GB against $26.9$ GB per GPU.

<figure style="display: flex; flex-direction: column; align-items: center; text-align: center;">
  <img src="{{ '/assets/images/mm_pretrain_speed_mem.png' | relative_url }}"
       alt="Training speed of the MultiMax run with separate clipping and a SoftMax run with identical settings on the same 192 GPUs, steps 48.2k to 66.8k"
       style="width:100%; max-width:820px; height:auto;">
  <figcaption style="max-width: 780px; margin: 0.6em auto; text-align: left;">
    <b>Figure 9:</b> Training speed on the same 192 GPUs, as a rolling median over 501 steps; the
    SoftMax run uses an MoE instead of a dense MLP in its MTP block. The MultiMax run resumes from a
    checkpoint at step 48k and reaches its steady rate by step 49.8k. The mean rates, which include
    occasional slow steps, are $0.3056$ and $0.3030$ steps per second.
  </figcaption>
</figure>

**The eight scalars follow their own training rules.** They are initialised to zero, where SegLU is
the identity; for lack of compute, the truncated normal initialisation (std 0.02) of the reference
implementation [2] was not tested at LLM scale and may give a further gain.

- **No weight decay.** At the learning rate of $6 \times 10^{-4}$ and weight decay of $0.1$ used
  below, decoupled decay alone would shrink a coefficient to about 1% of its value over 78k steps,
  back towards plain SoftMax.
- **Joint or separate clipping.** Each coefficient's gradient sums over the whole vocabulary, so the
  eight scalars dominate a shared gradient norm. Clipped together with the backbone, they raise the
  global norm from a median of $0.11$ in the baseline to $0.80$ from step 2k to 34k, and clipping
  rescales every gradient on 38% of these steps, against none in the baseline. They can instead be
  clipped in their own group at the same threshold; the next section runs both.
- **AdamW, not Muon.** Muon orthogonalises matrix-shaped weights, so the scalars go to AdamW.

## Pretraining a 10B-A1B MoE: lower loss over 600B tokens

This section compares two MultiMax runs on the vocabulary head with a SoftMax baseline over 600B
tokens.

**The three runs share the model and the recipe.** All train a 10B-A1B Latent Mixture-of-Experts
model [6], with 10B parameters in total and 1B active per token: 24 layers of hidden size 1,024 with
multi-head latent attention [7], 256 fine-grained routed experts of which eight are active per
token plus one shared expert [8], a vocabulary of 201,216 tokens and one MTP module as in
DeepSeek-V3 [7], with a dense MLP and a loss weight of $0.3$. Training runs for 78k steps at a
global batch of 1,920 sequences of 4,096 tokens in bfloat16, with Muon [9] for the matrix
parameters and AdamW for the rest, a learning rate of $6 \times 10^{-4}$ held constant after a
1,000-step warmup, weight decay $0.1$ and gradient clipping at $1.0$. In both MultiMax runs, SegLU
also modulates the shared MTP logits.

- **The run with joint clipping differs from the baseline only in the vocabulary head.** It starts
  from the same weights, its first loss matching the baseline's bit for bit, sees the same batches
  at every step, and clips the eight scalars together with the backbone.
- **The run with separate clipping gives the eight scalars their own gradient-norm group.** It starts
  from different initial weights (first loss $16.101$ against $16.088$), and from step 48k it
  resumes with a different data order. It also divides the coefficients' gradient by the square
  root of the vocabulary size. As AdamW ignores a constant scale of the gradient, this only
  keeps the coefficients from being clipped, which happens on 47% of steps without it; a run without
  it, on the same weights and batches, differs in mean loss by $0.0002$ from step 1k to 15.9k.

<figure style="display: flex; flex-direction: column; align-items: center; text-align: center;">
  <img src="{{ '/assets/images/mm_pretrain_loss.png' | relative_url }}"
       alt="Training loss over 78k steps for the SoftMax baseline and two MultiMax runs on the vocabulary head of a 10B-A1B MoE, and their differences from the baseline"
       style="width:100%; max-width:1000px; height:auto;">
  <figcaption style="max-width: 780px; margin: 0.6em auto; text-align: left;">
    <b>Figure 10:</b> Training loss of the 10B-A1B MoE over 78k steps (600B tokens), smoothed with an
    EMA of 0.999; it includes the MTP term, the language-modelling (LM) loss plus 0.3 × the MTP loss.
    <b>Left</b>: SoftMax baseline (grey) and MultiMax on the vocabulary head with separate clipping
    (red) and joint clipping (blue); the legend gives the mean over the last 1,000 steps.
    <b>Right</b>: each MultiMax run minus the baseline. On identical batches the difference is
    smooth: throughout for joint clipping, and up to step 48k (dotted line) for separate clipping,
    which then resumes with a different data order.
  </figcaption>
</figure>

**In both runs, the loss gap opens within the first 4k steps and holds** (Figure 10). With joint
clipping, it stays between $0.010$ and $0.012$ from step 40k on and is $0.011$ over the last 1,000
steps, $2.2529$ against $2.2636$; this run and the baseline continue on identical batches to step
97k (766B tokens), where the gap is $0.012$. With separate clipping, the gap is widest near step 8k,
at $0.019$, is $0.013$ at step 48k, the last step on identical batches, and stays between $0.010$
and $0.016$ from step 40k on. Over the last 1,000 steps, its loss is $2.2499$, a gap of $0.014$, the
same order as the gain above 0.01 on the 280B-A14B model; its LM loss is $1.6730$ and its MTP loss
$1.9228$, while the baseline logs only the total. The $0.003$ between the two MultiMax runs is not
attributed to the clipping, since the runs also differ in initial weights and data order.

## Reproducing the synthetic measurements

The script recomputes every synthetic measurement in this post and prints the main ones.

<details markdown="1">
<summary>Show the script (NumPy, about 40 lines)</summary>

```python
import numpy as np

K, rng = 2048, np.random.default_rng(0)
x = rng.normal(-2.0, 1.0, size=K)                       # long tail
mid = rng.choice(K, size=48, replace=False)             # keep these two lines in this
x[mid] = rng.uniform(0.5, 3.0, size=48)                 # order: the RNG stream matters
x[[0, 1, 2, 3]] = [6.0, 5.7, 5.4, 5.1]                  # four comparable modes

softmax = lambda z: np.exp(z - z.max()) / np.exp(z - z.max()).sum()
relu = lambda v: np.maximum(v, 0.0)

def sigma(x, t_b, t_d, b=0.0, d=4.5):                   # Eq. (3), first order
    return x + (1 - t_b) * relu(b - x) + (t_d - 1) * relu(x - d)

# eps and s: computed ONCE from the T=1 baseline, then frozen
p1 = softmax(x)
log_z = np.log(np.exp(x - x.max()).sum()) + x.max()
eps = np.log(1.0 / K) + log_z
relevant, tail = (x > eps) & (x < x.max()), x < eps     # 105 relevant, 1942 tail
imax = int(np.argmax(x))
s_ref = float(np.exp(np.log(p1[tail]).mean()))          # geometric mean of the baseline tail

def metrics(z):
    phi = softmax(z)
    M = 1.0 - (phi[imax] - phi[relevant]).mean()        # Def 3.2, Eq. (1)
    S = np.exp(-phi[tail] / s_ref).mean()               # Def 3.3, Eq. (2) simplified
    return float(M), float(S)

M0, S0 = metrics(x)
print("SoftMax T=1   ", (M0, S0))                                   # (0.7923, 0.3756)
print("MultiMax      ", metrics(sigma(x, t_b=3.0, t_d=0.3)))        # (0.8661, 0.8234)
print("term (1) only ", metrics(sigma(x, t_b=2.0, t_d=1.0)))        # (0.7625, 0.7276)

# Figure 2, right: which temperature reaches S = 0.8234, and at what M?
ts = np.concatenate([np.linspace(0.2, 1.0, 60), np.linspace(1.0, 6.0, 90)[1:]])
sweep = np.array([metrics(t * x) for t in ts])
print("SoftMax at the same S ->  T =", 1 / np.interp(0.8234, sweep[:, 1], ts),
      " M =", np.interp(0.8234, sweep[:, 1], sweep[:, 0]))   # T = 0.751, M = 0.6491

# Figure 4: share of the (t_b, t_d) grid that improves on T=1 in both metrics
grid = [(tb, td) for tb in np.linspace(1, 4, 60) for td in np.linspace(0.1, 1, 60)]
both = np.mean([all(np.greater(metrics(sigma(x, tb, td)), (M0, S0))) for tb, td in grid])
print("both improve on", round(100 * both, 1), "% of the grid")    # 74.5
```

</details>

## Limitations

The synthetic measurements use one multi-modal logit row: they demonstrate the mechanism and
reproduce Proposition 3.5 numerically, but make no claim about downstream quality, and the absolute
values of $M$ and $S$ depend on $\epsilon$ and $s$. Each pretraining configuration is a single run,
and the nanoGPT runs lack the training rules of the LLM implementation. The 10B-A1B runs keep a
constant learning rate, without a decay phase, which may change the gap. The run with separate
clipping also differs from the baseline in initial weights and, from step 48k, in data order. The
speed comparison rests on one allocation and a SoftMax run with a different MTP block.

### A note on paper versions

All references to [1] are checked against arXiv v3 (January 2025), not the ICML camera-ready
version.

- **Lemma 3.4.** In v1, v2 and the ICML version, the derivation behind Proposition 3.5 has a sign
  error, but its conclusion is correct. Lemma 3.4 states that sparsity increases with a tail
  probability $\phi(x)_l$, with the partial derivative given as $\frac{1}{1-s}\exp(\cdot) > 0$, so
  the proof of Proposition 3.5 ends with sparsity increasing in the temperature, while the
  proposition itself states, correctly, that sparsity decreases. v3 corrects the derivative to
  $-\frac{1}{s}\exp(\cdot) < 0$, which makes the lemma and the proof agree with the proposition; the
  statement of Proposition 3.5 and its condition are the same in all versions.
- **Figure 2 of [1].** The two-dimensional illustration scales only the varying entry, an error
  acknowledged in the [issue tracker](https://github.com/ZhouYuxuanYX/MultiMax/issues/2) in
  February 2025. It changes those curves but none of the conclusions, and it is not yet fixed in
  any arXiv version; every temperature here divides every entry.
- **FLOPs.** Section 4.4 of [1] quotes 0.0168 extra GFLOPs (0.37%) for DeiT-small; counting 15 FLOPs
  for each of the 2.8M attention entries gives about 0.04 GFLOPs (0.9%), the figure used here.

## Acknowledgements

Thanks to [Xudong Yan](https://github.com/demo4ai) for running the nanoGPT experiments.

## References

[1] Zhou, Yuxuan, Mario Fritz, and Margret Keuper. *MultiMax: Sparse and Multi-Modal Attention
Learning.* International Conference on Machine Learning (ICML), PMLR, 2024.
[arXiv:2406.01189v3](https://arxiv.org/abs/2406.01189v3)

[2] Reference implementation: [github.com/ZhouYuxuanYX/MultiMax](https://github.com/ZhouYuxuanYX/MultiMax)

[3] Touvron, Hugo, et al. *Training Data-Efficient Image Transformers & Distillation Through
Attention.* ICML, PMLR, 2021.

[4] Boyd, Stephen, Lin Xiao, and Almir Mutapcic. *Subgradient Methods.* Lecture notes, Stanford
University, 2003.

[5] Karpathy, Andrej. *nanoGPT.* [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
— GPT-2 (124M) on OpenWebText, `train.py` defaults, reference val loss ~2.85.

[6] Blakeman, Aaron, et al. *NVIDIA Nemotron 3: Efficient and Open Intelligence.* 2025.
[arXiv:2512.20856](https://arxiv.org/abs/2512.20856)

[7] Liu, Aixin, et al. *DeepSeek-V3 Technical Report.* 2024.
[arXiv:2412.19437](https://arxiv.org/abs/2412.19437)

[8] Dai, Damai, et al. *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts
Language Models.* Annual Meeting of the Association for Computational Linguistics (ACL), 2024.

[9] Jordan, Keller, et al. *Muon: An Optimizer for Hidden Layers in Neural Networks.* 2024.
[kellerjordan.github.io/posts/muon](https://kellerjordan.github.io/posts/muon/)

<!-- MathJax for the equations of this post: the site theme (minima 2.5) does not load it -->
<script>
  window.MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ['\\[', '\\]']],
      processEscapes: true
    },
    options: { skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'] }
  };
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

<!-- Page views at the bottom right (vercount, a busuanzi-compatible counter); hidden until the count loads -->
<div><span id="vercount_container_page_pv" style="display: none; position: fixed; right: 14px; bottom: 12px; z-index: 20; font-size: 12px; color: #6b7280; background: rgba(255, 255, 255, 0.92); border: 1px solid #e5e7eb; border-radius: 4px; padding: 2px 8px;"><span id="vercount_value_page_pv"></span> views</span></div>
<script defer src="https://events.vercount.one/js"></script>
