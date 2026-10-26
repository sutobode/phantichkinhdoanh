# Executive Brief — Customer Retention & Promotion Targeting

**To:** Marketing / CRM decision-maker | **Evidence:** UCI Online Retail II,
5,281 customers in the final 90-day temporal test beginning 2011-09-10.

## Decision and main finding

**Do not launch any of the three currently assumed discount campaigns.** None
offers positive simulated NET margin after fixed-unit product costs and contact
expense. **No Promotion = £0 simulated incremental margin** is the decision reference.
An available budget does not have to be spent.

The real-data baseline predicts repeat purchase with PR-AUC **0.750184** and
ROC-AUC **0.789977** (actual repeat rate **43.36%**; naive PR-AUC **0.433630**).
Future-gross-sales MAE is **£395.69**, RMSE **£2,016.14**. These are temporal-test
prediction metrics, not evidence that promotion causes additional sales.

Advanced modeling strengthens but does not change the decision: GBM classification
improves final-test PR-AUC to **0.769056**, while regression accuracy is temporally
mixed across five secondary origins. The baseline remains the interpretable business
reference; monitoring is more appropriate than changing promotion assumptions.

## Policy comparison — simulated NET margin

| Scenario | Non-targeted mean | Random-policy 95% interval | RFM | Proposed / No Promotion |
|---|---:|---|---:|---:|
| Conservative | -£17,417.59 | [-£21,707.94, -£14,023.76] | -£55,333.45 | £0 / £0 |
| Moderate | -£37,407.57 | [-£46,583.42, -£30,081.21] | -£118,314.35 | £0 / £0 |
| Aggressive | -£90,677.15 | [-£112,076.51, -£73,360.59] | -£277,865.97 | £0 / £0 |

Non-targeted averages **1,000 seeded random campaigns**, each contacting **1,500**
customers; its interval describes variation across random customer selections,
**not a causal confidence interval**. RFM also contacts 1,500. Both spend **£750**
at **£0.50/contact**. Proposed selects only positive-NET-margin customers, up to
the same capacity; it selects zero and spends £0 in all three scenarios.
Source: `outputs/scenario_table.csv`.

## Assumptions and target groups

- Discounts/base sales-volume growth: **5%/10%, 10%/20%, 20%/35%**. Gross margin:
  **40%**. Growth is an assumed proportional increase, not conversion probability.
- Segment growth multipliers: Champions 0.5, Loyal 1.0, At-risk 1.3, Lost 0.4;
  these are business assumptions, not measured treatment responsiveness.
- Unit product costs remain fixed. NET contribution is
  `v*((1+r)*(g-d)-g) - £0.50`, with nonnegative predicted gross sales `v`.
- **Nurture:** positive NET contribution, ranked under the budget. None currently
  qualify. **Retain:** positive-value Champions/Loyal without discounted offers;
  any service/relationship benefit is unmeasured. **No-offer:** remaining customers.

## Sensitivity and decision gate

At 10% discount, positive-value At-risk customers need median assumed base growth
of **26.75%**, exceeding the configured 20%; their median supportable discount at
20% growth is **7.91%**. Explore smaller discounts or independently supported
growth assumptions rather than changing assumptions merely to produce profit.

First profitable points on the tested discount × growth grid are **5%/20%**
(£7,175.92), **10%/30%** (£1,424.27), and **15%/50%** (£962.39). No campaign is
profitable at 20% discount through 50% growth. In separate one-at-a-time checks,
35% growth at 10% discount gives £3,990.80. These are exploratory, non-causal
scenarios—not evidence to launch a campaign. Validate any changed assumption
through a randomized pilot before rollout.

## Limitations

There is no randomized treatment/control: no causal uplift, true promotion ROI,
or actual incremental profit is claimed. Forecasts describe historical gross
positive sales, not untreated revenue, net returns or lifetime CLV. Of 1,067,371
raw rows, 34,335 exact duplicates are retained under the documented source policy.
Outliers, return exclusions, time drift and uncertain growth assumptions limit
transfer to current customers. Detailed methodology and provenance are in the
model card and `outputs/run_manifest.json`; the current evaluation score artifact
is `data/processed/customer_test_scores.csv`.
Removing 34,335 exact duplicate rows in a separate sensitivity leaves 5,281 customers
and the same no-campaign recommendation; it does not establish that duplicates are errors.
