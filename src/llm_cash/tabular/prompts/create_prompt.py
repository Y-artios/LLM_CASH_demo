from llm_cash.tabular.prompts.utils import json_to_markdown
from typing import Dict, Any, Iterable, List, Union, Optional
from pathlib import Path
import json
import os



def create_hyperparameter_grid(
    models_list: Iterable[str],
    models_dir: Optional[Path],
    filename: str = "hyperparameter_grid.json",
) -> Dict[str, Any]:
    """
    Load per-model hyperparameter grids and return a combined mapping:
        {
          "<model_name>": {
            "<param>": {"dtype": "<str>", "values": [...]},
            ...
          },
          ...
        }


    Args:
        models_list: iterable of model folder names (e.g. ["catboost", "lgbm", "xgboost", "skmlp"]).
        models_dir: Optional path to the models directory.
        filename: JSON file to read inside each model folder.

    Returns:
        dict mapping model name -> its grid dict 
    """

    if not models_dir:
        models_root = Path(__file__).resolve().parent.parent 
    else:
        models_root = models_dir 

    combined = {}

    for model in models_list:
        grid_path = models_root / model / filename
        if not grid_path.is_file():
            raise FileNotFoundError(f"Missing grid file for '{model}': {grid_path}")

        with grid_path.open("r") as f:
            grid = json.load(f)

        if not isinstance(grid, dict):
            raise ValueError(
                f"Grid file for '{model}' must be a JSON object (dict), got {type(grid).__name__}."
            )

        # No transformation: keep the exact shape you showed in your example
        combined[model] = grid

    return combined


def grid_to_models_format(grid: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert a combined hyperparameter grid:
        {
          "<model>": {
            "<param>": {"dtype": "...", "values": [...]},
            ...
          },
          ...
        }
    into:
        {
          "models": {
            "<model>": {
              "columns": ["<param1>", "<param2>", ...],
              "values": []
            },
            ...
          }
        }
    """
    models: Dict[str, Any] = {}
    for model_name, params in grid.items():
        columns = list(params.keys())
        models[model_name] = {"columns": columns, "values": []}
    return {"models": models}


def create_user_prompt(current_dataset: str, meta_datasets: Iterable[str], models_list: Iterable[str], datasets_dir:Optional[Path], models_dir:Optional[Path],) -> str:
    """
    Build a textual prompt for the language model that describes the current task,
    past tasks, and the hyperparameter search space.

    The prompt combines:
      1. A Markdown view of the full hyperparameter grid.
      2. Metadata and top model results from previous datasets of the same task type
         (classification or regression).
      3. A detailed description of the current dataset.
      4. (If zero-shot) A JSON format example showing the structure expected for the
         model configuration output.

    Parameters
    ----------
    current_dataset : str
        Name of the dataset for the current task (without file extension).
    meta_datasets : Iterable[str]
        Names of past datasets to include as examples.
        If empty, this triggers the zero-shot setting.
    models_list : Iterable[str]
        List of model names to include in the hyperparameter grid.
    datasets_dir : Optional[Path]
        Directory containing "datasets" subdirectory with dataset metadata.
    models_dir : Optional[Path]
        Directory containing ""models subdirectory with hyperparameter grids.

    Returns
    -------
    str
        The full user prompt containing all information and instructions.
    """
    hyperparameter_grid = create_hyperparameter_grid(models_list, models_dir=models_dir)
    hyperparameter_grid_md = json_to_markdown(hyperparameter_grid)

    prompt_text = (
        "Here is the hyperparameter grid:\n"
        f"{hyperparameter_grid_md}\n"
    )
    if not datasets_dir:
        datasets_root = Path(__file__).resolve().parent.parent 
    else:
        datasets_root = datasets_dir 


    with open(datasets_root / current_dataset / "metadata.json", "r") as f:
        metadata = json.load(f)
        
    

    if "regression" in metadata["dataset"]["prediction_type"]:
        task_type = "regression"
    else:
        task_type = "classification"

    if meta_datasets:
        prompt_text += "Here are the descriptions of the past tasks with their best performing models:\n"
        for name in meta_datasets:
            if name != current_dataset:
                metadata_path = datasets_root / name /"metadata.json"
                models_path = datasets_root / name /"top_10_submissions.json"
                
                if not os.path.isfile(models_path):
                    print(f"Warning: Missing models file for dataset '{name}': {models_path}. Skipping.")
                    continue
            
                with open(metadata_path, "r") as f:
                    meta_json = json.load(f)
                
                with open(models_path, "r") as f:
                    models = json.load(f)

                if task_type in meta_json["dataset"]["prediction_type"]:
                    prompt_text += json_to_markdown(meta_json) + "\n" + str(models) + "\n"


    prompt_text += "\nHere is a description of the new task:\n" + json_to_markdown(metadata)

    if not meta_datasets:
        # Zero-shot setting needs format example
        json_format = grid_to_models_format(hyperparameter_grid)
        json_format_str = json.dumps(json_format, indent=2)
        prompt_text += "\nHere is the format you must complete:\n" + json_format_str

    return prompt_text



if __name__== "__main__":

    models = ["catboost", "lgbm", "xgboost", "skmlp"]

    test_prompt = create_user_prompt(
        current_dataset="kaggle_abalone",  # replace with real one, e.g., "house_prices"
        meta_datasets=["kaggle_blueberry", "kaggle_cirrhosis"],  # or [] for zero-shot
        models_list=models,
    )
    
    print(test_prompt)
