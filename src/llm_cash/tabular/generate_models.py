import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from llm_cash.tabular.prompts.create_prompt import create_user_prompt

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
        



def main():
    parser = argparse.ArgumentParser(
        description="Generate and validate model configurations using an LLM."
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Name of the task to process."
    )
    parser.add_argument(
        "--zero-shot",
        action="store_true",
        help="Use zero-shot prompting (no other tasks in context). Default: meta-informed mode."
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="deepseek-reasoner",
        help="Name of the LLM model to query. Default: deepseek-reasoner."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. (Default: 42)"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Directory containing 'models/' subdirectory with model definitions and hyperparameter grids. (Default: uses built-in models)"
    )
    parser.add_argument(
        "--tasks-dir",
        type=str,
        default=None,
        help="Directory containing 'tasks/' subdirectory with tasks and metadata. (Default: uses built-in tasks)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory where generated files will be saved. (Default: current directory)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for the LLM. (Default: 1.0)"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL for the LLM API endpoint (useful for custom or self-hosted gateways)."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the LLM provider. If omitted, reads from the OPENAI_API_KEY environment variable."
    )
    parser.add_argument(
        "--save-reasoning",
        action="store_true",
        help="Save model reasoning content (if available) to 'reasoning.txt'."
    )
    parser.add_argument(
        "--save-prompt",
        action="store_true",
        help="Save the user prompt sent to the model as 'user_prompt.txt'."
    )
    args = parser.parse_args()
    
    root_dir = Path(__file__).resolve().parent
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    models_root = (
        Path(args.models_dir) / "models"
        if args.models_dir and (Path(args.models_dir) / "models").exists()
        else root_dir / "models"
    )

    tasks_root = (
        Path(args.tasks_dir) / "tasks"
        if args.tasks_dir and (Path(args.tasks_dir) / "tasks").exists()
        else root_dir / "tasks"
    )

    print(f"Using models directory:   {models_root}")
    print(f"Using tasks directory: {tasks_root}")
    
    tasks_list = [p.name for p in tasks_root.iterdir() if p.is_dir()] 
    models_list = [p.name for p in models_root.iterdir() if p.is_dir()]

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")

    client_kwargs = {}
    if args.base_url:
        client_kwargs['base_url'] = args.base_url
        print(f"  Using base URL: {args.base_url}")
    if api_key:
        client_kwargs['api_key'] = api_key
        print(f"  Using provided API key.")    
    if not client_kwargs:
        print("  Warning: No base_url or api_key provided, using default OpenAI config")
    
    client = OpenAI(**client_kwargs)


    if args.zero_shot:
        tasks_list = []
        with open(root_dir / "prompts" / "zero_shot_system_prompt.txt") as f:
            sys_prompt = f.read()
    else:
        with open(root_dir / "prompts" / "meta_informed_system_prompt.txt") as f:
            sys_prompt = f.read()
    
    user_prompt = create_user_prompt(args.task, tasks_list, models_list, tasks_dir=tasks_root, models_dir=models_root)
    
    if args.save_prompt:
        with open(output_dir / "user_prompt.txt", "w") as f:
            f.write(user_prompt)
        print("Reasoning saved to 'reasoning.txt'")

    response = client.chat.completions.create(
            model=args.llm_model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            seed = args.seed,
            stream = False,
            temperature = args.temperature
        )
    
    if args.save_reasoning:
        reasoning = getattr(response.choices[0].message, "reasoning_content", None)
        if reasoning:
            with open(output_dir / "reasoning.txt", "w") as f:
                f.write(reasoning)
            print("Reasoning saved to 'reasoning.txt'")
        else:
            print("No reasoning content found; skipping save.")

    models = extract_json(response.choices[0].message.content)

    with open(output_dir / "models.json", "w") as f:
        json.dump(models, f, indent=4)
    print("Models saved to 'models.json'.")

if __name__ == "__main__":
    main()
