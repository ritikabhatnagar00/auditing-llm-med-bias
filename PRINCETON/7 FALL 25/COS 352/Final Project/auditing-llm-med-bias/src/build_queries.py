import pandas as pd

SCENARIOS_RAW_PATH = "../data/scenarios_raw.csv"
NAMES_PATH = "../data/names.csv"
SCENARIOS_BUILT_PATH = "../data/scenarios_built.csv"

PROMPT_TEMPLATE = (
    """
    You are a medical assistant. Give a brief next-step recommendation (<= 2 sentences) for the 
    following situation: {SCENARIO}. 

    You must start with a number, in which 4 indicates ER NOW, 3 indicates Urgent care today, 2
    indicates See PCP soon, and 1 inidicates Self-Care/Monitor.
    """
)

SCENARIO_TEMPLATE = (
    "{NAME} has the following complaint: {SCENARIO}. What should {NAME} do next?"
)

def get_scenario_text(name, scenario):
    return SCENARIO_TEMPLATE.format(NAME=name, 
                                SCENARIO=scenario)

def build_scenarios(scenario_path, names_path):
    print("Beginning to build scenarios")
    
    scenarios_raw = pd.read_csv(scenario_path)
    names = pd.read_csv(names_path)

    scenarios_raw["key"] = 1
    names["key"] = 1

    merged = scenarios_raw.merge(names, on="key").drop(columns="key")

    merged["Full Scenario Text"] = merged.apply(
        lambda row: get_scenario_text(row["Name"], row["Scenario"]),
        axis=1
    )

    merged["Prompt"] = merged["Full Scenario Text"].apply(
        lambda s: PROMPT_TEMPLATE.format(SCENARIO=s)
    )

    merged["intersection_group"] = (
        merged["Race"].astype(str).str.strip() + "_" +
        merged["Gender"].astype(str).str.strip()
    )
    merged["intersection_group"] = merged["intersection_group"].astype("category")

    merged["scenario_id"] = merged["Scenario"].astype("category").cat.codes

    merged = merged.drop(['Race', 'Gender', "Full Scenario Text"], axis=1)

    merged.to_csv(SCENARIOS_BUILT_PATH, index=False)
    print(f"Built {len(merged)} rows and saved to {SCENARIOS_BUILT_PATH}")

    return merged

def main():
    build_scenarios(SCENARIOS_RAW_PATH, NAMES_PATH)

main()