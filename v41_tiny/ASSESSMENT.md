## Assessment

**v4.1 is a real partial improvement over v4 at this scale, but the tested
concept system has not earned a claim of learned hierarchical reasoning.**
Across three pinpad seeds, adding the retrieval package raises held-out action
accuracy by 5.55 percentage points and pair-second accuracy by 16.67 points.
It closes about 42% of the measured v4-to-transformer action gap. The transformer
still leads by 7.76 action points and 12.28 pair-second points at the matched
token budgets. Language loss improves slightly across both text seeds. These
are useful capability gains, conditional on the tasks, widths and budgets used.

The fully replicated controls support keeping the private decoder for the
next study: the decoder stack lowers action accuracy by 3.27 points on average.
Removing the repeated encoder F pass changes pair-second accuracy by +0.72
points, with 0.87 points of paired item SE and 1.90 points of training-seed SE.
It fails the registered keep criterion while saving about 18% of the estimated
pinpad forward FLOPs. Use one encoder F pass in the next variant; its trained
text guard still needs checking. Keep concept-core pass count separate from
that encoder decision: trained M=1 was not tested here.

The transformer and two stack seeds are still improving at the final budget.
The registered comparison is therefore budget-limited; the measured ranking
does not establish a converged or full-size ranking.

From first principles, this makes sense. The model needs access to the current
cell, a relevant earlier colour, the relation between colours and the target's
location. R supplies a direct read of earlier token features, relieving the
compressed concept route of carrying every distant fact. Its causal mask and
private-plan boundary work as specified. However, an information route that
exists need not carry a representation that improves decisions.

The calibrated tests expose that distinction. Replacing v4.1's example-specific
plans with one training-derived mean lowers pair-second accuracy by only 1.61
points, versus 11.28 points in v4 without R. The effect is small rather than
literally zero. On text, the same replacement costs 0.040 nats, so plans have
more visible utility there. Nonconstant vectors, active gradients and weakly
readable task information should not be mistaken for either a completely idle
core or proof of useful abstract reasoning.

The strongest warning concerns the validation gate. The learned header null
is a different numerical input from a normal body plan. A model trained without
reading any concept content still suffers a large `core_off` drop when that
null is inserted. Thus the proposed switch-off can produce a false positive
for route use. This gate needs calibration before it can justify a large
architecture decision.

The matched GPU training control is also worth following up: removing decoder
plan content during training improves action accuracy by 6.74 points and
pair-second accuracy by 11.17 points in seed 0. It also follows changed facts
better. This is a post-hoc, one-seed result, and auxiliary concept learning
remains present. It supports testing a simpler or gated plan injection; it
does not establish that all concept training is useless.

The frozen oracle provides encouraging control evidence: legal C offsets repair
25 of 52 initial errors at the largest tested learning rate without damaging
the 76 initially correct cases. This uses correct labels, full-width offsets
and one-step decisions. A learned low-dimensional controller, useful switching
and long-horizon control remain untested.
No matched transformer-residual oracle was tested, so steerability alone does
not establish a unique concept-interface advantage.

My score gives substantial credit to the precise information boundaries,
implementation checks, inexpensive capability gain and steerable latent site.
It deducts for the remaining transformer gap, weak generated-policy structure,
limited evidence for useful example-specific abstractions and the invalid
route-attribution gate. It is a judgment of the tested design, not a prediction
that a larger model must get the same score.

The next worthwhile study should replicate normal versus trained plan-off
across seeds on pinpad and text, then compare a gated plan injection. Keep R
as a demonstrated candidate. Require fact-following, calibrated plan utility
and generated task structure before treating the concept interface as learned
planning. The fully replicated stack and encoder-loop decisions are reported
in the [report](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/REPORT.md)
using the original rules.
