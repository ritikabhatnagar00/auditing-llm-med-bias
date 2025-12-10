import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

pd.options.mode.chained_assignment = None

def edit_df(df):
    # Add correctness flag
    df["correct"] = (df["urgency_score"] == df["Ground truth"]).astype(int)

    df_clean = df.dropna(subset=["urgency_score"])
    df_wrong = df_clean[df_clean["urgency_score"] != df_clean["Ground truth"]]

    # Add over- and under-estimation metrics
    def classify_error(row):
        if row["urgency_score"] > row["Ground truth"]:
            return "over"
        else:
            return "under"
    df_wrong["error_direction"] = df_wrong.apply(classify_error, axis=1)
    df_wrong["error_under"] = (df_wrong["error_direction"] == "under").astype(int)

    # Add minimization language flag
    def get_min_flag(val):
        if isinstance(val, tuple) and len(val) == 2:
            return bool(val[0])

        if isinstance(val, bool):
            return val
        
        return False

    df["min_flag"] = df["has_minimization"].apply(get_min_flag)
    
    # Add safety violation flag
    def is_missing_val(val):
        if pd.isna(val):
            return True
        if isinstance(val, str) and val.strip() == "":
            return True
        return False

    df["is_missing"] = df["urgency_score"].apply(is_missing_val)
    df["is_safety_violation"] = df["is_safety violation"]
    
    return df, df_wrong

def get_overall_accuracy_by_group(df, group_by="intersection_group", prediction="urgency_score"):
    df = df.dropna(subset=[prediction])

    group_acc = (
        df.groupby(group_by)["correct"]
        .agg(
            n="count",
            accuracy="mean"
        )
    )

    print("Accuracy by demographic group:")
    print(group_acc)

def get_over_under_by_group(df, group_by="intersection_group", prediction="error_direction"):

    # Count over/under-estimations per group
    error_stats = (
        df
        .groupby([group_by, prediction])
        .size()
        .unstack(fill_value=0)
    )

    for col in ["over", "under"]:
        if col not in error_stats.columns:
            error_stats[col] = 0

    # Add totals and percentages
    error_stats["total_wrong"] = error_stats["over"] + error_stats["under"]
    error_stats["pct_over"] = error_stats["over"] / error_stats["total_wrong"]
    error_stats["pct_under"] = error_stats["under"] / error_stats["total_wrong"]

    print(error_stats[["over", "under", "total_wrong", "pct_over", "pct_under"]])

def count_minimization_language(df, group_by="intersection_group", prediction="min_flag"):
    min_summary = (
        df.groupby(group_by)[prediction]
        .agg(
            n_total="count",
            n_minimization="sum"
        )
        .reset_index()
    )

    min_summary["pct_minimization"] = min_summary["n_minimization"] / min_summary["n_total"]
    print(min_summary)

def count_safety_violations(df, group_by="intersection_group", prediction="is_safety violation"):
    safety_summary = (
        df.groupby(group_by)
        .agg(
            n_rows=(prediction, "size"),
            n_missing=("is_missing", "sum"),
            n_safety_violation=(prediction, "sum"),
        )
    )

    print(safety_summary)

def detect_bias(df, dependent_var, group_by="intersection_group", reference_group="White_Male"):

    order = ["White_Male", "White_Female", "Black_Female", "Black_Male"]

    df[f"{group_by}"] = (
        df[f"{group_by}"]
        .astype("category")
        .cat.reorder_categories(order, ordered=False)
    )

    formula = f"{dependent_var} ~ C({group_by})"
    
    model = smf.gee(
        formula,
        groups="scenario_id",
        data=df,
        family=sm.families.Binomial()
    )
    results = model.fit()
    print(results.summary())

def main():
    model = "medalpaca"
    df = pd.read_csv(f"../data/scenarios_scored/scenarios_scored_{model}.csv")
    df, df_wrong = edit_df(df)

    # Accuracy and significance testing
    get_overall_accuracy_by_group(df)
    print("-----")
    detect_bias(df, "correct")
    print("-------------------")

    # Under-estimation and signifcance
    get_over_under_by_group(df_wrong)
    print("-----")
    detect_bias(df_wrong, "error_under")
    print("--------------------")
    
    # Minimization lanaguage and significance
    count_minimization_language(df)
    print("-----")
    detect_bias(df_wrong, "has_minimization")
    print("--------------------")

    # # Safety violations and significance
    count_safety_violations(df)
    print("-----")
    detect_bias(df, "is_safety_violation")
    print("--------------------")


main()

