import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_DIR = "/scratch/network/rb5539/hf_models/medalpaca-7b"

INPUT_CSV  = os.path.join("../data/scenarios_built.csv")
OUTPUT_CSV = os.path.join("../data/scenarios_built_medalpaca.csv")

PROMPT_COLUMN = "Prompt"

def main():
    
    # Load scenarios 
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Cannot find input CSV: {INPUT_CSV}")
    
    df = pd.read_csv(INPUT_CSV)
    if PROMPT_COLUMN not in df.columns:
        raise KeyError(
            f"Column '{PROMPT_COLUMN}' not found in {INPUT_CSV}. "
            f"Available columns: {list(df.columns)}"
        )

    print("Loading tokenizer and model from:", MODEL_DIR, flush=True)

    # Set up model pipeline from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    # Run model over each prompt
    outputs = []
    for idx, row in df.iterrows():
        user_prompt = str(row[PROMPT_COLUMN])
        
        prompt_text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
            ### Instruction:
            {user_prompt}

            ### Response:
        """
        
        result = gen_pipe(
            prompt_text,
            max_new_tokens=64,
            do_sample=False,
            temperature=0.0,
        )
        
        full_text = result[0]["generated_text"]
        completion = full_text[len(prompt_text):].strip()
        outputs.append(completion)
        
        if (idx + 1) % 10 == 0:
            print(f"Completed {idx + 1}/{len(df)} prompts...", flush=True)

    # Save outputs
    df["medalpaca_output"] = outputs
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved Med-Alpaca outputs to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()