# Executive Brief — Customer Retention & Promotion Targeting

**To:** Marketing / CRM decision-maker | **Data:** UCI Online Retail II historical transactions

## Decision

Prioritize customers only when a promotion has positive **simulated NET margin**
after discount-related product costs and contact expense. A no-campaign decision
is permitted; an available budget does not have to be spent.

## Current recommendation

**Do not launch any of the three currently assumed discount campaigns.** Keeping
unit product costs fixed changes the earlier economic conclusion. Even for the
highest assumed-growth segment (At-risk), before-contact incremental margin is:

| Scenario | Discount | Base sales-volume growth | Best coefficient × predicted revenue |
|---|---:|---:|---:|
| Conservative | 5% | 10% | -0.0045 |
| Moderate | 10% | 20% | -0.0220 |
| Aggressive | 20% | 35% | -0.1090 |

Subtracting £0.50/contact makes every customer contribution negative when predicted
revenue is nonnegative. Each proposed policy therefore contacts **0 customers**, costs
**£0** and produces **£0 simulated NET change**. Equal-contact random and RFM
comparators also contact zero; their differences are zero. These are consequences
of the stated assumptions, not observed campaign results.

The former positive-margin recommendations and customer counts are **superseded**.
They used different economics and mixed in-sample predictions. Corrected UCI
predictive metrics/population counts await a raw-data rerun; no model accuracy or
positive campaign return is claimed here.

## Policy and target groups

- **Nurture:** positive predicted revenue AND positive NET scenario margin;
  rank descending and select within the contact budget. None qualify under the
  three current assumptions; sensitivity can explore genuinely different assumptions.
- **Retain:** positive-value Champions/Loyal with nonpositive promotional margin;
  consider non-discount relationship/service attention. Its benefit is not estimated.
- **No-offer:** remaining customers; no discounted campaign recommended.

Contact budget is **£750**, at **£0.50/customer**, maximum **1,500**. Comparisons use
the same evaluation population, actual contact count, cost and assumptions:
seeded random versus RFM ranking versus positive-NET-margin ranking.

## Evidence and assumptions

- The required baseline is logistic repeat-purchase prediction plus linear
  future-gross-sales prediction and RFM. GBM remains an optional comparison.
- Features are known before each decision. Train origins: 2010-12-01 and 2011-03-01;
  validation: 2011-06-01; final test: 2011-09-10. Each outcome lasts 90 days.
  Models are refitted only on fully observed earlier labels before final prediction.
- Main business outputs use final temporal-test predictions only, not demo scores.
- Value means predicted **gross positive sales under historical behavior**, clipped
  at zero; not net returns, lifetime CLV or an identified no-promotion counterfactual.
- Gross margin is assumed 40%. Unit product costs stay unchanged with discount.
  NET contribution is `v*((1+r)*(g-d)-g) - 0.50`.
- `response_rate` means assumed proportional pre-discount sales-volume growth,
  not conversion probability. Segment multipliers are Champions 0.5, Loyal 1.0,
  At-risk 1.3, Lost 0.4; they are unmeasured business assumptions.
- No randomized promotion treatment/control exists. This is scenario analysis,
  **not causal uplift, true promotion ROI or actual incremental profit**.

## Next decision gate and risks

Complete the raw-data rerun before using predictive metrics. Review the four-axis
sensitivity table and test any revised promotion assumption through a randomized
pilot before rollout. Monitor realized product margin, returns and contact costs.
Historical sales, retained duplicates, outlier predictions and time drift limit
transfer to current customers. Do not replace assumptions solely to obtain a
profitable recommendation.
