import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from prompts.create_prompt import create_user_prompt

from openai import OpenAI

def extract_json(text: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Extract the first valid JSON object from a model response."""
    json_str: Optional[str] = None

    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        json_str = fenced.group(1)
    else:
        brace = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        if brace:
            json_str = brace.group(1)

    if not json_str:
        print("No JSON object found in output.")
        return None

    # Fix malformed decimals
    json_str = re.sub(r"(?<![\d])\.(\d)", r"0.\1", json_str)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON extracted: {e}")
        return None

def validate_models(json_data: Dict[str, Any], hyperparameter_grid: Dict[str, Any]) -> bool:
    """Validate that generated model configurations match the expected hyperparameter grid."""
    if not isinstance(json_data, dict) or "models" not in json_data:
        print("Invalid format: missing top-level 'models' key.")
        return False

    total = sum(len(m.get("values", [])) for m in json_data["models"].values())
    if total != 10:
        print(f"Expected 10 models total, but found {total}.")
        return False

    for model_name, model_data in json_data["models"].items():
        if model_name not in hyperparameter_grid:
            print(f"Model '{model_name}' not found in the hyperparameter grid.")
            return False

        grid_params = hyperparameter_grid[model_name]
        columns = model_data.get("columns", [])
        values = model_data.get("values", [])

        if len(columns) != len(grid_params):
            print(f"Model '{model_name}': column count mismatch.")
            return False

        for c in columns:
            if c not in grid_params:
                print(f"Model '{model_name}': column '{c}' not in hyperparameter grid.")
                return False

        for row in values:
            if len(row) != len(columns):
                print(f"Model '{model_name}': row length mismatch.")
                return False
            for i, col in enumerate(columns):
                val = row[i]
                if val not in grid_params[col]["values"]:
                    print(f"Model '{model_name}': value '{val}' not allowed for '{col}'.")
                    return False

    print("Models configuration validated successfully.\n")
    return True
        



def main():
    parser = argparse.ArgumentParser(description="Run LLM generation.")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name.")
    parser.add_argument("--zero-shot", type=bool, default=False,
                        help="Whether to use zero-shot or meta-informed (default: False).")
    parser.add_argument("--llm_model", type=str, default="deepseek-r1",
                        help="LLM model name (default: deepseek-r1).")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature for LLM calls (default: 1.0).")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory to save outputs.")
    parser.add_argument("--base-url", type=str, default=None,
                        help="Base URL for the LLM API endpoint (e.g., OpenAI-compatible gateway).")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key for the LLM provider. If omitted, uses environment variable or default client.")
    parser.add_argument("--save-reasoning", type=bool, default=False,
                        help="Whether to save reasoning outputs (default: False).")
    parser.add_argument("--save-prompt", type=bool, default=False,
                        help="Whether to save user prompt (default: False).")
    
    args = parser.parse_args()
    
    root_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir)

    models_root = root_dir / "models"
    datasets_root = root_dir / "datasets"
    datasets_list = [p.name for p in datasets_root.iterdir() if p.is_dir()]
    models_list = [p.name for p in models_root.iterdir() if p.is_dir()]


    if not args.api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    else:
        api_key = args.api_key

    client = OpenAI(api_key=api_key, base_url=args.base_url)


    if args.zero_shot:
        datasets_list = []
        with open(root_dir / "prompts" / "zero_shot_system_prompt.txt") as f:
            sys_prompt = f.read()
    else:
        with open(root_dir / "prompts" / "meta_informed_system_prompt.txt") as f:
            sys_prompt = f.read()
    
    user_prompt = create_user_prompt(args.dataset, datasets_list, models_list)
    
    if args.save_prompt:
        with open(output_dir / "user_prompt.txt", "w") as f:
            f.write(user_prompt)
        print("Reasoning saved to 'reasoning.txt'")

    response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            seed = args.seed,
            stream = False,
            temperature = args.temperature
        )
    
    if args.save_reasoning:
        with open(output_dir / "reasoning.txt", "w") as f:
            f.write(response.choices[0].message.reasoning_content)
        print("Reasoning saved to 'reasoning.txt'")

    models = extract_json(response.choices[0].message.content)


    with open(output_dir / "models.json", "w") as f:
        json.dump(models, f, indent=4)
    print("Models saved to 'models.json'.")

if __name__ == "__main__":
    main()
