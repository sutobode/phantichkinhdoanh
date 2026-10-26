# BA Capstone — Phase 4 Robustness + Final Submission Audit

Work from the CURRENT repository state.

The project already has:
- corrected temporal validation;
- baseline Logistic Regression / Linear Regression / RFM;
- final-test-only business simulation;
- No Promotion reference;
- Monte Carlo non-targeted benchmark;
- break-even promotion analysis;
- response × discount sensitivity surface;
- Scenario Sandbox dashboard;
- reproducibility manifest;
- GBM challenger models;
- calibration diagnostics;
- SHAP attribution;
- rolling-origin temporal robustness;
- passing final QA and tests.

## Objective

Complete ONLY:

1. **Duplicate-policy robustness analysis**
2. **Final end-to-end submission audit and cleanup**

Do NOT add new model families.
Do NOT tune models.
Do NOT redesign the business economics.
Do NOT change the primary target.
Do NOT change temporal dates.
Do NOT change the current business recommendation merely to make results more attractive.

Stop when the acceptance criteria are met.

---

# Scope discipline

Inspect only files needed for robustness and submission QA.

Start with:

- `.gitignore`
- `README.md`
- `requirements.txt`
- `config/params.yaml`
- `src/02_cleaning_features.py`
- `src/03_baseline_models.py`
- `src/05_decision_scenario.py`
- `src/06_final_qa.py`
- `dashboard/app.py`
- `tests/`
- `docs/problem_statement.md`
- `docs/data_dictionary.md`
- `docs/model_card.md`
- `docs/executive_brief.md`
- `docs/traceability_map.md`
- `slides/outline.md`
- `outputs/run_manifest.json`
- `outputs/data_quality.csv`
- `outputs/scenario_table.csv`
- `outputs/break_even_summary.csv`
- `outputs/response_discount_surface.csv`
- `outputs/temporal_model_comparison.csv`
- `outputs/model_selection.csv`
- `outputs/calibration_metrics.csv`
- `outputs/rolling_origin_metrics.csv`
- `models/temporal_registry.json`

Read other files only if directly referenced and required.

Do NOT perform an open-ended refactor or architecture review.

---

# Current validated facts

Preserve these unless the existing generated artifacts prove otherwise:

- raw rows: `1,067,371`
- exact duplicate rows beyond first: `34,335`
- main duplicate policy: `retain`
- final temporal-test customers: `5,281`
- final-test Logistic PR-AUC: approximately `0.7502`
- final-test Logistic ROC-AUC: approximately `0.7900`
- final-test Linear MAE: approximately `£395.69`
- final-test Linear RMSE: approximately `£2,016.14`
- final business prediction source: `baseline`
- current three configured promotion scenarios: Proposed selects zero customers
- current recommendation: do not launch the configured campaigns under stated assumptions

The current main target remains `future_gross_sales`.

Do NOT replace it with a net-sales target in this task.

---

# PART A — Duplicate-policy robustness

## 1. Purpose

The raw dataset contains many exact duplicate rows.

The main pipeline currently retains them because the source does not establish whether identical transaction lines are erroneous duplicates or legitimate repeated line items.

We need a SECONDARY robustness analysis:

```text
Main analysis: retain exact duplicates
Sensitivity analysis: remove exact duplicate rows
```

The sensitivity result must NOT silently replace the primary analysis.

---

## 2. Minimal implementation approach

Prefer the smallest implementation that avoids destabilizing the main pipeline.

Acceptable patterns:

- a dedicated robustness script;
- a configurable cleaning override used only by the robustness run;
- a small helper reused by the robustness script.

Do NOT duplicate the whole pipeline unless unavoidable.

Do NOT change the default duplicate policy from `retain`.

Do NOT modify primary generated outputs during the robustness run if that risks confusion.

Use clearly separated robustness artifacts, for example:

```text
outputs/duplicate_robustness_summary.csv
outputs/duplicate_robustness_model_metrics.csv
outputs/duplicate_robustness_business_metrics.csv
```

A single compact summary file is also acceptable if cleanly designed.

---

## 3. What to compare

At minimum compare `retain` versus `drop_exact_duplicates` on:

### Data
- transaction rows after duplicate handling;
- final-test customer population;
- repeat-purchase prevalence if affected.

### Baseline predictive performance
- Logistic PR-AUC;
- Logistic ROC-AUC;
- Linear MAE;
- Linear RMSE.

Use the SAME temporal design and final test.

### Business conclusion
At minimum compare:

- Proposed contacts for Conservative / Moderate / Aggressive;
- Proposed simulated NET margin;
- non-targeted Monte Carlo mean if practical;
- whether the main campaign recommendation changes.

If rerunning the full Monte Carlo benchmark is expensive, use the existing configured number of simulations unless runtime becomes unreasonable. Do not reduce it silently.

### Interpretation

Report whether the main conclusions are:

- quantitatively stable;
- qualitatively stable;
- materially changed.

Define a simple, transparent interpretation instead of inventing a complex statistical test.

For example:

> The main conclusion is qualitatively robust if the selected campaign/no-campaign decision is unchanged across duplicate policies.

Do not claim exact duplicates are confirmed errors.

---

## 4. Tests for robustness logic

Add only focused tests.

At minimum test:

1. default main pipeline still retains exact duplicates;
2. robustness mode removes exact duplicates beyond the first occurrence;
3. duplicate handling affects only the robustness path;
4. robustness run preserves the same temporal cutoffs;
5. robustness artifacts clearly label the duplicate policy;
6. main outputs are not overwritten by the sensitivity run.

Do not create redundant tests.

---

# PART B — Final submission audit

Audit the project against the course requirements.

This is a submission-quality audit, not a new modeling task.

---

## 5. Reproducibility / run instructions

Verify README instructions are sufficient for a clean machine.

The expected high-level flow should be understandable and executable:

```bash
python -m venv .venv
# activate environment
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/02_cleaning_features.py
python src/03_baseline_models.py
python src/04_advanced_models.py      # optional/stretch if documented as such
python src/05_decision_scenario.py
python src/06_final_qa.py
streamlit run dashboard/app.py
```

Do not make advanced models mandatory if the project is designed so baseline-only business analysis works.

Ensure Python 3.11 is stated consistently.

Verify `pip check` passes.

---

## 6. Data provenance

Check that the project documents:

- source URL / dataset identity;
- download date if known;
- raw-data SHA-256;
- raw row count;
- major filters;
- cancellation/return treatment;
- duplicate policy;
- observation/outcome logic;
- joins if any.

Do not invent a download date if none is known.

If unavailable, explicitly mark it as unavailable rather than fabricating it.

---

## 7. Deliverable consistency audit

Check all current deliverables for stale or contradictory numbers.

Audit:

- `README.md`
- `docs/problem_statement.md`
- `docs/data_dictionary.md`
- `docs/model_card.md`
- `docs/executive_brief.md`
- `docs/traceability_map.md`
- `slides/outline.md`
- dashboard captions / labels

Verify consistency for at least:

- final-test population;
- temporal dates;
- repeat-purchase metrics;
- value-model metrics;
- Monte Carlo benchmark;
- break-even findings;
- scenario recommendation;
- advanced model comparison;
- calibration decision;
- rolling-origin interpretation;
- duplicate policy;
- causal/non-causal wording.

Do NOT rewrite prose broadly if it is already correct.

Fix only real inconsistencies, stale numbers, misleading wording, or missing required evidence.

---

## 8. Executive brief audit

The executive brief must remain suitable for a manager and reasonably fit the course maximum of 2 pages under standard formatting.

It should prioritize:

1. business decision;
2. key recommendation;
3. evidence;
4. assumptions;
5. decision boundary / break-even insight;
6. limitations.

Do not turn it into a technical report.

Move technical detail to the model card if necessary.

---

## 9. Model card audit

Ensure the model card includes:

- unit of analysis;
- targets;
- features / feature families;
- temporal validation design;
- baseline models;
- optional advanced challengers;
- model-selection rules;
- final-test discipline;
- calibration;
- SHAP interpretation caveat;
- business prediction source;
- assumptions;
- risks;
- ethics/bias considerations;
- limitations.

Explicitly state that SHAP is predictive attribution, not causality.

Explicitly state promotion response is scenario-based, not estimated treatment effect.

---

## 10. Dashboard audit

Verify the dashboard actually supports the course requirement of:

- 3–5 decision-relevant KPIs;
- actionable view;
- segment/value context;
- promotion scenario outcomes;
- Scenario Sandbox;
- explicit non-causal warning;
- No Promotion = £0 reference.

Smoke-test:

```bash
streamlit run dashboard/app.py
```

If practical, use the existing health/AppTest checks.

Do not redesign the dashboard if it already satisfies these requirements.

---

## 11. Git / artifact hygiene

Check:

```bash
git status --short
git diff --check
```

Ensure:

### Should generally be tracked
- source code;
- tests;
- docs;
- small decision-relevant CSV outputs;
- final customer score table;
- figures used in deliverables;
- model registry;
- run manifest.

### Should generally stay ignored
- raw source workbook;
- virtual environments;
- Python caches;
- large intermediate processed datasets;
- serialized `.joblib` / `.pkl` models unless the project specifically requires shipping them.

Check that legacy artifacts are not presented as current outputs.

In particular, ensure documentation consistently points to:

```text
data/processed/customer_test_scores.csv
```

as the current final customer-level score table.

---

## 12. Causal-language audit

Search affected docs/dashboard/slides for wording such as:

- causal uplift;
- revenue lift;
- ROI;
- savings;
- treatment effect;
- campaign caused;
- expected observed increase

Correct only if the wording incorrectly implies causal identification.

Preferred language:

- simulated incremental margin;
- assumed response;
- scenario-based;
- decision scenario;
- avoided simulated loss relative to benchmark.

Do not weaken valid predictive statements.

---

## 13. Final QA enhancement

Update `src/06_final_qa.py` only if useful to check submission-critical invariants not already covered.

Potential checks:

- final score table exists;
- run manifest exists;
- current scenario table exists;
- Monte Carlo columns present;
- break-even outputs present;
- response-discount surface present;
- advanced model comparison outputs present;
- duplicate robustness artifact present;
- final temporal dates match config;
- business prediction source is baseline.

Keep QA maintainable.

Do not duplicate all test logic inside QA.

---

# PART C — Verification

Run the appropriate pipelines for any changed logic.

At minimum:

```bash
python src/02_cleaning_features.py
python src/03_baseline_models.py
python src/05_decision_scenario.py
python src/06_final_qa.py
python -m pytest tests/ -q
pip check
git diff --check
```

Run the duplicate robustness analysis.

Do NOT rerun Phase 3 advanced models unless:
- required by documentation consistency;
- the robustness implementation explicitly compares advanced models;
- or final QA normally invokes them.

Smoke-test the dashboard.

If any command cannot run, state exactly why.

---

# Acceptance criteria

The task is complete when:

## Duplicate robustness
- main policy remains `retain`;
- a clearly separated `drop_exact_duplicates` sensitivity exists;
- predictive/business outcomes are compared;
- primary outputs are not overwritten;
- conclusion stability is explicitly reported.

## Submission package
- README is executable and consistent;
- data provenance is clear;
- current outputs are referenced everywhere;
- executive brief is concise;
- model card is complete;
- dashboard passes smoke-test;
- slides contain no stale results;
- causal language is disciplined;
- final score table is tracked/current;
- legacy artifacts are not presented as authoritative.

## Reproducibility
- Python 3.11 remains standard;
- `pip check` passes;
- run manifest remains valid;
- tests pass;
- final QA passes;
- `git diff --check` passes.

## Preservation
- no temporal redesign;
- no new model family;
- no post-hoc model switching;
- no changes to economics to manufacture profit;
- no unnecessary refactor.

Stop once these criteria are satisfied.

---

# Final response format

Return only:

## Duplicate robustness

Provide a compact retain-vs-drop comparison:
- row counts;
- final population;
- PR-AUC / ROC-AUC;
- MAE / RMSE;
- Proposed contacts / margin for each scenario;
- whether the main recommendation changes.

## Submission audit fixes

Only actual fixes made.

## Files changed

Exact paths.

## Verification

- duplicate robustness run;
- pytest;
- final QA;
- pip check;
- git diff --check;
- dashboard smoke-test.

## Remaining limitations

Only substantive remaining limitations.

## Final readiness

State one of:

- `READY FOR FINAL SUBMISSION`
- `READY WITH MINOR MANUAL PACKAGING`
- `NOT READY`

and give the minimum reason.

Do not provide a broad repository walkthrough.
