# Initial LightGBM error diagnosis

Analysis uses saved development-validation predictions.
No models were retrained and the published evaluation split was not used.

## Findings

LightGBM performs worse than the seven-day average in both stockout
segments in fold 1. It improves both segments in fold 2. In fold 3,
it improves the stockout-affected segment but worsens the other segment.

Stockout segments refer to the recorded 06:00–22:00 window.
These observational comparisons do not establish causality.

LightGBM improves 104, 114, and 111 of 200 series in folds 1, 2, and 3,
respectively. A majority of series improving does not guarantee lower
pooled error because improvements and losses differ in magnitude.

Large fold-1 losses include both underprediction of a higher-volume
series and overprediction of a low-volume series.

## Next experiment

Evaluate one fixed 50:50 blend of LightGBM and the seven-day average.

This tests whether retaining part of the baseline prediction reduces
LightGBM's larger deviations. It does not assume the blend will improve.

Use identical validation rows and metrics. Do not search for an optimal
blend weight on these results.

Any improvement remains a development-validation finding, requiring
final assessment on the reserved evaluation split.