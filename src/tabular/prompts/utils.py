import json
from typing import List, Union, Dict, Any


def concatenate_model_json_files(models_list: List[str]) -> Dict:
    """
    Concatenate multiple JSON files containing model information into a single JSON object.
    
    Each JSON file should have a structure similar to:
    {
        "models": {
            "model_name": {
                "columns": [...],
                "values": [...]
            },
            ...
        }
    }
    
    Parameters:
    models_list (list): List of model JSON files to concatenate
    
    Returns:
    dict: Combined JSON object with all model data
    """    
    # Initialize the result structure
    combined_data = {"models": {}}
    
    # Process each file
    for data in models_list:
            
        # Check if the file has the expected structure
        if "models" not in data:
            print(f"Warning: File does not have the expected 'models' key. Skipping.")
            continue
            
        # Process each model in the file
        for model_name, model_data in data["models"].items():
            # If this model doesn't exist in the combined data yet, add it
            if model_name not in combined_data["models"]:
                combined_data["models"][model_name] = {
                    "columns": model_data.get("columns", []),
                    "values": model_data.get("values", [])
                }
                
            else:
                # Check if columns match
                existing_columns = combined_data["models"][model_name]["columns"]
                new_columns = model_data.get("columns", [])
                
                if existing_columns != new_columns:
                    print(f"Warning: Columns for model {model_name} don't match between files.")
                    raise(ValueError("Column mismatch"))
                
                # Add values from this file to the existing model
                combined_data["models"][model_name]["values"].extend(model_data.get("values", []))
                
    return combined_data
                        
                        
def json_to_markdown(json_data: Union[Dict, List, str], title: str = "JSON Data") -> str:
    """
    Convert JSON data to Markdown format.
    
    Args:
        json_data: JSON data as a dictionary, list, or string
        title: Title for the Markdown document
        
    Returns:
        Markdown formatted string
    """
    # Parse JSON string if needed
    if isinstance(json_data, str):
        try:
            json_data = json.loads(json_data)
        except json.JSONDecodeError:
            return "Error: Invalid JSON string provided."
    
    # Start with the title
    markdown = f"# {title}\n\n"
    
    # Process the JSON data
    if isinstance(json_data, dict):
        markdown += _process_dict(json_data)
    elif isinstance(json_data, list):
        markdown += _process_list(json_data)
    else:
        markdown += f"{json_data}\n"
    
    return markdown

def _process_dict(data: Dict, level: int = 0) -> str:
    """Process a dictionary and convert it to markdown."""
    result = ""
    for key, value in data.items():
        # Add header based on nesting level
        header = "#" * (level + 2) if level < 4 else "######"
        
        if isinstance(value, dict):
            result += f"{header} {key}\n\n"
            result += _process_dict(value, level + 1)
        elif isinstance(value, list):
            result += f"{header} {key}\n\n"
            result += _process_list(value, level + 1)
        else:
            if level == 0:
                result += f"{header} {key}\n\n{value}\n\n"
            else:
                result += f"**{key}**: {value}\n\n"
    
    return result

def _process_list(data: List, level: int = 0) -> str:
    """Process a list and convert it to markdown."""
    result = ""
    
    # Check if it's a list of dictionaries
    if data and all(isinstance(item, dict) for item in data):
        # Create a table if possible
        if len(data) > 0:
            # Get all possible keys
            all_keys = set()
            for item in data:
                all_keys.update(item.keys())
            
            # Table header
            header = "| " + " | ".join(all_keys) + " |\n"
            separator = "| " + " | ".join(["---" for _ in all_keys]) + " |\n"
            
            # Table rows
            rows = ""
            for item in data:
                row = "| "
                for key in all_keys:
                    cell = str(item.get(key, "")) if item.get(key, "") is not None else ""
                    row += cell + " | "
                rows += row + "\n"
            
            result += header + separator + rows + "\n"
        return result
    
    # Regular list
    for item in data:
        if isinstance(item, dict):
            result += "- " + _process_dict(item, level + 1).replace("\n", "\n  ").strip() + "\n"
        elif isinstance(item, list):
            result += "- " + _process_list(item, level + 1).replace("\n", "\n  ").strip() + "\n"
        else:
            result += f"- {item}\n"
    
    result += "\n"
    return result
