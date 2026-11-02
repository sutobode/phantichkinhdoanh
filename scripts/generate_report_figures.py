"""Generate report-ready figures from current BA capstone artifacts.

Usage (copy to <repo>/scripts/generate_report_figures.py):
    python scripts/generate_report_figures.py
    python scripts/generate_report_figures.py --only timeline,segments,policy
    python scripts/generate_report_figures.py --list

Outputs PNG (240 dpi) and SVG to outputs/report_figures/.
No modeling, source-data mutations, assumptions changes, or network calls.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURES = (
    "workflow", "timeline", "value_distribution", "segments", "cohort",
    "model_ap", "model_mae", "policy", "random_intervals", "break_even",
)
HISTORY_START = pd.Timestamp("2009-12-01")  # verified in config/params.yaml
FINAL_HISTORY_END_EXCLUSIVE = pd.Timestamp("2011-09-10")
SOURCE_FILES = {
    "workflow": [],
    "timeline": ["outputs/temporal_schedule.csv"],
    "value_distribution": ["data/processed/customer_test_scores.csv"],
    "segments": ["data/processed/customer_test_scores.csv"],
    "cohort": ["outputs/cohort_retention.csv"],
    "model_ap": ["outputs/temporal_model_comparison.csv"],
    "model_mae": ["outputs/temporal_model_comparison.csv"],
    "policy": ["outputs/scenario_table.csv"],
    "random_intervals": ["outputs/scenario_table.csv"],
    "break_even": ["outputs/break_even_by_segment.csv", "outputs/scenario_table.csv"],
}


def read_csv(repo: Path, relative: str, columns: tuple[str, ...]) -> pd.DataFrame:
    path = repo / relative
    if not path.is_file():
        raise FileNotFoundError(f"Required source not found: {path}")
    df = pd.read_csv(path)
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"Source {relative} missing columns: {missing}; observed {list(df.columns)}")
    return df


def save(fig: plt.Figure, dest: Path, name: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest / f"{name}.png", dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(dest / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def workflow(repo: Path, out: Path) -> str:
    stages = [
        ("Transaction data", "Online Retail II\nline-item history"),
        ("Customer snapshots", "Cleaning, RFM/cohorts\ndecision-time features"),
        ("Predictive analytics", "Repeat purchase and\n90-day gross-sales forecasts"),
        ("Business economics", "Discount × assumed growth\nmargin and contact cost"),
        ("Decision support", "Budget-constrained targeting\nNo Promotion comparator"),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ys = np.linspace(0.89, 0.14, len(stages))
    for i, ((header, detail), y) in enumerate(zip(stages, ys)):
        ax.text(0.5, y, f"{header}\n{detail}", ha="center", va="center", fontsize=11,
                linespacing=1.35, bbox={"boxstyle":"round,pad=0.7", "linewidth":1.1})
        if i < len(ys) - 1:
            ax.annotate("", xy=(0.5, ys[i+1] + 0.075), xytext=(0.5, y - 0.075),
                        arrowprops={"arrowstyle":"->", "linewidth":1.6})
    ax.set_title("Business-first analytics and decision framework", fontsize=14, pad=20)
    save(fig, out, "F01_analytics_workflow")
    return "F01_analytics_workflow"


def timeline(repo: Path, out: Path) -> str:
    df = read_csv(repo, "outputs/temporal_schedule.csv",
                  ("decision_date", "outcome_end_exclusive", "split"))
    df["decision_date"] = pd.to_datetime(df["decision_date"])
    df["outcome_end_exclusive"] = pd.to_datetime(df["outcome_end_exclusive"])
    df = df.sort_values("decision_date").reset_index(drop=True)
    if len(df) != 4:
        raise ValueError("Main temporal schedule must contain exactly 4 rows (2 train, validation, test).")
    fig, ax = plt.subplots(figsize=(12, 4.6))
    left = mdates.date2num(HISTORY_START)
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, row in df.iterrows():
        y = len(df) - i - 1
        decision = mdates.date2num(row["decision_date"])
        end = mdates.date2num(row["outcome_end_exclusive"])
        ax.barh(y, decision-left, left=left, height=.42, hatch="//", color=palette[0], label="Observed history" if i==0 else None)
        ax.barh(y, end-decision, left=decision, height=.42, hatch="xx", color=palette[1], label="Future outcome (90 days)" if i==0 else None)
        ax.plot([decision, decision], [y-.28,y+.28], linewidth=1.5)
    ax.set_yticks(range(3,-1,-1), ["Train 1", "Train 2", "Validation", "Final test"])
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_xlim(HISTORY_START-pd.Timedelta(days=20),df["outcome_end_exclusive"].max()+pd.Timedelta(days=10))
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    ax.set_title("Expanding observation histories and future outcome windows")
    ax.set_xlabel("Transaction calendar date (half-open windows)")
    fig.tight_layout()
    save(fig, out, "F05_temporal_timeline")
    return "F05_temporal_timeline"


def value_distribution(repo: Path, out: Path) -> str:
    df = read_csv(repo, "data/processed/customer_test_scores.csv", ("future_gross_sales",))
    sales = pd.to_numeric(df["future_gross_sales"], errors="raise")
    if sales.isna().any() or (sales < 0).any():
        raise ValueError("Observed future_gross_sales should be finite and nonnegative")
    positives = sales[sales > 0]
    if positives.empty:
        raise ValueError("No positive sales; cannot create a logarithmic value histogram")
    fig, ax = plt.subplots(figsize=(9.4, 5.5))
    bounds = np.logspace(np.log10(positives.min()),np.log10(positives.max()),40)
    ax.hist(positives,bins=bounds)
    ax.set_xscale("log")
    ax.set_title("Observed next-90-day gross sales (final temporal test)")
    ax.set_xlabel("Observed positive gross sales per customer (£, log scale)")
    ax.set_ylabel("Customers")
    ax.text(.98,.96,f"N = {len(sales):,}\nZero sales = {(sales==0).sum():,} ({(sales==0).mean():.1%})",
            transform=ax.transAxes,ha="right",va="top",bbox={"boxstyle":"round,pad=.35"})
    fig.tight_layout()
    save(fig,out,"F02_future_gross_sales_distribution")
    return "F02_future_gross_sales_distribution"


def segments(repo: Path, out: Path) -> str:
    df = read_csv(repo, "data/processed/customer_test_scores.csv", ("rfm_segment",))
    order = ["Champions", "Loyal", "At-risk", "Lost"]
    counts = df["rfm_segment"].value_counts()
    unexpected = set(counts.index) - set(order)
    if unexpected:
        raise ValueError(f"Unexpected RFM segment labels: {unexpected}")
    vals = [int(counts.get(s,0)) for s in order]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.bar(order,vals)
    ax.bar_label(bars,padding=4)
    ax.set_ylim(0,max(vals)*1.18)
    ax.set_title("RFM segment sizes at the final decision date")
    ax.set_ylabel("Unique customers")
    ax.set_xlabel("RFM rule-based segment")
    fig.tight_layout()
    save(fig,out,"F03_rfm_segment_population")
    return "F03_rfm_segment_population"


def cohort(repo: Path, out: Path) -> str:
    df = read_csv(repo,"outputs/cohort_retention.csv",
                  ("cohort_month","months_since_acquisition","retention_rate"))
    df["cohort_month"] = pd.to_datetime(df["cohort_month"]).dt.to_period("M")
    df["months_since_acquisition"] = pd.to_numeric(df["months_since_acquisition"],errors="raise").astype(int)
    df["retention_rate"] = pd.to_numeric(df["retention_rate"],errors="raise")
    if ((df["retention_rate"]<0)|(df["retention_rate"]>1)).any():
        raise ValueError("retention_rate must be within [0,1] fractions")
    last_full_month = FINAL_HISTORY_END_EXCLUSIVE.to_period("M")-1
    # Last observed September 2011 is partial. Do not treat it as a complete-month rate.
    df = df[df.apply(lambda r:r["cohort_month"]+int(r["months_since_acquisition"])<=last_full_month,axis=1)]
    if df.empty:
        raise ValueError("No completed cohort months available")
    df["cohort_month"] = df["cohort_month"].astype(str)
    matrix = df.pivot_table(index="cohort_month",columns="months_since_acquisition",
                            values="retention_rate",aggfunc="first").sort_index()
    fig,ax=plt.subplots(figsize=(12,7))
    arr = np.ma.masked_invalid(matrix.to_numpy(dtype=float))
    im=ax.imshow(arr,aspect="auto",interpolation="nearest",vmin=0,vmax=1)
    ax.set_yticks(np.arange(len(matrix)))
    ax.set_yticklabels(matrix.index,fontsize=8)
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns.astype(str),fontsize=8)
    ax.set_xlabel("Months since first observed purchase")
    ax.set_ylabel("First observed purchase cohort")
    ax.set_title("Observed monthly cohort activity (completed months only)")
    fig.colorbar(im,ax=ax,label="Share active during the month")
    fig.tight_layout()
    save(fig,out,"F04_cohort_activity_heatmap")
    return "F04_cohort_activity_heatmap"


def _model_table(repo:Path,task:str,metric:str)->pd.DataFrame:
    specs={
        "classification": ("classifier", {
            "logistic_regression": "Baseline",
            "gradient_boosting_classifier": "Gradient Boosting",
        }),
        "regression": ("regressor", {
            "linear_regression": "Baseline",
            "gradient_boosting_regressor": "Gradient Boosting",
        }),
    }
    if task not in specs:
        raise ValueError(f"Unsupported model-comparison task: {task}")
    estimator_col,estimators=specs[task]
    df=read_csv(repo,"outputs/temporal_model_comparison.csv",
                ("split","model",estimator_col,metric)).copy()
    def split(value:str)->str:
        s=str(value).lower()
        if "valid" in s: return "Validation"
        if "final" in s or "test" in s: return "Final Test"
        return "Unknown"
    df["estimator"] = df[estimator_col].astype(str).str.lower()
    df["family"] = df["estimator"].map(estimators)
    df["period"] = df["split"].map(split)
    df=df[df.family.notna() & df.period.isin(["Validation","Final Test"])]
    df[metric]=pd.to_numeric(df[metric],errors="raise")
    if df[metric].isna().any() or not np.isfinite(df[metric]).all():
        raise ValueError(f"{task} comparison contains missing or non-finite {metric} values")
    keys=["family","period"]
    duplicates=df[df.duplicated(keys,keep=False)]
    if not duplicates.empty:
        conflicts=duplicates[["split","model",estimator_col,metric]].to_dict("records")
        raise ValueError(f"Ambiguous {task} comparison rows for {keys}: {conflicts}")
    expected={(family,period) for family in ("Baseline","Gradient Boosting")
              for period in ("Validation","Final Test")}
    observed=set(df[keys].itertuples(index=False,name=None))
    if observed!=expected:
        missing=sorted(expected-observed)
        extra=sorted(observed-expected)
        raise ValueError(f"Incomplete {task} comparison; missing keys={missing}, unexpected keys={extra}")
    return df


def _metric_chart(repo:Path,out:Path,task:str,metric:str,stem:str,label:str,
                  title:str,higher_is_better:bool)->str:
    df=_model_table(repo,task,metric)
    series=[]
    for family in ("Baseline","Gradient Boosting"):
        vals=[]
        for period in ("Validation","Final Test"):
            val=df.loc[(df.family==family)&(df.period==period),metric]
            vals.append(float(val.iloc[0]))
        series.append((family,vals))
    x=np.arange(2)
    fig,ax=plt.subplots(figsize=(8,4.8))
    for i,(family,vals) in enumerate(series):
        bars=ax.bar(x+(i-.5)*.34,vals,width=.34,label=family)
        ax.bar_label(bars,fmt="%.3f" if metric=="pr_auc" else "£%.2f",padding=3,fontsize=9)
    ax.set_xticks(x,["Validation","Final Test"])
    ax.set_ylabel(label)
    direction="Higher is better" if higher_is_better else "Lower is better"
    ax.set_title(f"{title}\n{direction}")
    ax.legend()
    ax.set_ylim(0,1 if metric=="pr_auc" else max(max(v) for _,v in series)*1.18)
    fig.tight_layout()
    save(fig,out,stem)
    return stem


def model_ap(repo:Path,out:Path)->str:
    return _metric_chart(repo,out,"classification","pr_auc",
                         "F06a_repeat_purchase_ap_comparison","Average precision (PR-AUC)",
                         "Repeat purchase classification on temporal evaluation",True)


def model_mae(repo:Path,out:Path)->str:
    return _metric_chart(repo,out,"regression","mae",
                         "F06b_future_sales_mae_comparison","MAE (£ per customer / 90 days)",
                         "Future gross sales regression on temporal evaluation",False)


def _scenario_table(repo:Path)->pd.DataFrame:
    df=read_csv(repo,"outputs/scenario_table.csv",
                  ("scenario","policy","simulated_net_incremental_margin","customers_targeted"))
    # Strong guard against accidentally plotting the old pre-Monte-Carlo score table.
    required=("random_policy_simulations","random_mean_net_margin","random_p025_net_margin","random_p975_net_margin")
    missing=[c for c in required if c not in df]
    if missing:
        raise ValueError("Current Monte Carlo scenario output required; missing " + ", ".join(missing)
                         + ". Do NOT plot historical pre-Monte-Carlo artifacts.")
    return df


def policy(repo:Path,out:Path)->str:
    df=_scenario_table(repo)
    scenarios=["conservative","moderate","aggressive"]
    policies=["non_targeted","rfm","proposed"]
    if not set(scenarios).issubset(set(df.scenario.str.lower())):
        raise ValueError("Scenario table is missing one or more current configured scenarios")
    x=np.arange(len(scenarios))
    fig,ax=plt.subplots(figsize=(11,5.6))
    for k,p in enumerate(policies):
        values=[]
        for s in scenarios:
            row=df.loc[(df.scenario.str.lower()==s)&(df.policy.str.lower()==p)]
            if len(row)!=1:
                raise ValueError(f"Expected one scenario/policy row for {s}/{p}")
            values.append(float(row.iloc[0].simulated_net_incremental_margin))
        label={"non_targeted":"Non-targeted (MC mean)","rfm":"RFM","proposed":"Proposed"}[p]
        if p == "proposed" and np.allclose(values, 0):
            ax.plot(x+(k-1)*.25, values, linestyle="none", marker="o", markersize=7,
                    label="Proposed: 0 contacts / £0")
            for xx in x+(k-1)*.25:
                ax.annotate("£0", (xx, 0), xytext=(0, -15), textcoords="offset points",
                            ha="center", fontsize=9)
        else:
            ax.bar(x+(k-1)*.25,values,width=.25,label=label)
    ax.axhline(0,linestyle="--",linewidth=1.3,label="No Promotion = £0")
    ax.set_ylim(float(df["simulated_net_incremental_margin"].min())*1.1, 45000)
    ax.set_xticks(x,[s.capitalize() for s in scenarios])
    ax.set_ylabel("Simulated incremental NET margin (£)")
    ax.set_title("Campaign policies under three configured economic scenarios")
    ax.legend(loc="lower left")
    ax.text(.99,.98,"Same maximum capacity K=1,500; Proposed may abstain\nAll values simulated, not causal",
            transform=ax.transAxes,ha="right",va="top",fontsize=9,
            bbox={"boxstyle":"round,pad=.35"})
    fig.tight_layout()
    save(fig,out,"F09_policy_scenario_comparison")
    return "F09_policy_scenario_comparison"


def random_intervals(repo:Path,out:Path)->str:
    df=_scenario_table(repo)
    df=df[df.policy.str.lower()=="non_targeted"].copy()
    order=["conservative","moderate","aggressive"]
    df["scenario"] = df["scenario"].str.lower()
    df=df.set_index("scenario").loc[order]
    for col in ("random_mean_net_margin","random_p025_net_margin","random_p975_net_margin"):
        df[col]=pd.to_numeric(df[col],errors="raise")
    mean=df.random_mean_net_margin.to_numpy(dtype=float)
    low=df.random_p025_net_margin.to_numpy(dtype=float)
    high=df.random_p975_net_margin.to_numpy(dtype=float)
    if not np.all((low<=mean)&(mean<=high)):
        raise ValueError("Monte Carlo intervals do not contain their means")
    fig,ax=plt.subplots(figsize=(9.6,4.8))
    y=np.arange(3)
    ax.errorbar(mean,y,xerr=np.vstack([mean-low,high-mean]),fmt="o",capsize=5,linewidth=2)
    ax.axvline(0,linestyle="--",label="No Promotion = £0")
    ax.set_yticks(y,[s.capitalize() for s in order])
    ax.invert_yaxis()
    ax.set_xlabel("Simulated NET margin (£)")
    ax.set_title("Non-targeted campaign selection variation (1,000 random draws)")
    ax.legend()
    ax.text(.99,.04,"2.5th–97.5th sample percentiles; not a causal CI",
            transform=ax.transAxes,ha="right",fontsize=9)
    fig.tight_layout()
    save(fig,out,"F09b_random_policy_simulation_intervals")
    return "F09b_random_policy_simulation_intervals"


def break_even(repo:Path,out:Path)->str:
    df=read_csv(repo,"outputs/break_even_by_segment.csv",
                ("scenario","segment","required_base_response_median"))
    scenarios=read_csv(repo,"outputs/scenario_table.csv",("scenario","response_rate_reference"))
    assumed_row=scenarios.loc[scenarios.scenario.str.lower()=="moderate","response_rate_reference"]
    if assumed_row.empty: raise ValueError("Moderate scenario absent")
    assumed=float(assumed_row.iloc[0])*100
    df=df[df.scenario.str.lower()=="moderate"].set_index("segment")
    order=["Champions","Loyal","At-risk","Lost"]
    missing=[s for s in order if s not in df.index]
    if missing: raise ValueError(f"Missing segment break-even rows: {missing}")
    vals=pd.to_numeric(df.loc[order,"required_base_response_median"],errors="raise").to_numpy()*100
    fig,ax=plt.subplots(figsize=(8.4,5))
    bars=ax.barh(order,vals)
    ax.bar_label(bars,fmt="%.1f%%",padding=5)
    ax.axvline(assumed,linestyle="--",linewidth=1.4,label=f"Configured base growth: {assumed:g}%")
    ax.set_xlim(0,max(vals.max()*1.2,assumed*1.3))
    ax.invert_yaxis()
    ax.set_xlabel("Median break-even assumed base sales growth (%)")
    ax.set_title("Break-even growth by RFM segment — Moderate scenario\n"
                 "Medians among customers with positive forecast value")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig,out,"F10_break_even_growth_by_segment")
    return "F10_break_even_growth_by_segment"


PLOTTERS={name:globals()[name] for name in FIGURES}


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,default=Path(__file__).resolve().parent.parent,
                        help="Repository root (default assumes this file is placed under scripts/)")
    parser.add_argument("--out",type=Path,default=None,
                        help="Output dir (default <repo>/outputs/report_figures)")
    parser.add_argument("--only",default=",".join(FIGURES),
                        help="Comma-separated chart keys; use --list to view keys")
    parser.add_argument("--list",action="store_true")
    parser.add_argument("--strict",action="store_true",help="Fail immediately on missing source/invalid artifact")
    args=parser.parse_args()
    if args.list:
        for name in FIGURES:
            print(f"{name:20} {', '.join(SOURCE_FILES[name]) or '[conceptual]'}")
        return
    repo=args.repo.resolve()
    out=(args.out.resolve() if args.out else repo/"outputs"/"report_figures")
    selected=[x.strip() for x in args.only.split(",") if x.strip()]
    unknown=set(selected)-set(FIGURES)
    if unknown: parser.error("Unknown figure keys: "+", ".join(sorted(unknown)))
    plt.rcParams.update({"font.size":10,"axes.titlesize":12,"axes.labelsize":10})
    results=[]
    for name in selected:
        try:
            stem=PLOTTERS[name](repo,out)
            print(f"OK     {stem}.png / .svg")
            results.append((name,stem,"generated","; ".join(SOURCE_FILES[name])))
        except (FileNotFoundError,ValueError,KeyError,TypeError) as exc:
            print(f"SKIP   {name}: {exc}")
            results.append((name,"","skipped: "+str(exc),"; ".join(SOURCE_FILES[name])))
            if args.strict: raise
    if results:
        out.mkdir(parents=True,exist_ok=True)
        with (out/"figure_generation_status.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.writer(f)
            w.writerow(["key","file_stem","status","source_files"])
            w.writerows(results)
    print(f"Completed: {sum(r[2]=='generated' for r in results)}/{len(results)} figure groups; output: {out}")


if __name__=="__main__":
    main()
