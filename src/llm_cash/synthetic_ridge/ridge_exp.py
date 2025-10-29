import json, argparse, os
from typing import List, Optional
from textwrap import dedent
import pandas as pd
import random
import re
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from llm_cash.synthetic_ridge.tasks import Task, sample_task
from llm_cash.synthetic_ridge.best_lambda import LAMBDA_GRID
import numpy as np

baselines = ["mean", "logistic-classifier"]

num_regex = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')

def extract_lambda(text):
    """
    Return float value if the first numeric token in *text* parses,
    else None.
    """
    m = num_regex.search(text)
    if m:
        try:
            return float(m.group())
        except ValueError:
            pass          # fall through, return None
    return None


def create_prompt(current_task: "Task", context_tasks: List["Task"]) -> str:
    """
    Build a single-string prompt for the LLM:
      • Declares the common lambda-grid once.
      • Lists past tasks with their known optimal lambda.
      • Presents the new task without lambbda_star.
      • Ends with a clear instruction to output one value.
    """


    if not(context_tasks):
        new_entry = current_task.metadata(include_lambda=False, task_id="0")
        
        prompt_parts = [
        "You are a statistics assistant. "
        "Your task is to inspect a Gaussian-classification problem that will be solved with ridge regression "
        "and then pick the optimal ridge-regularisation constant lambda for this problem (task_id: 0). ",
        "The task is a two-class Gaussian problem with:",
        "  • n1, n2   : sample counts for classes 1 and 2;",
        "  • mu1, mu2 : mean vectors of the two classes;", 
        "  • alpha1, alpha2 : AR(1) Toeplitz correlation coefficients",
        "    defining each class's covariance Sigma_ij = alpha^{|i-j|}.",
        "Choose lambda only from the common grid provided below. ",
        "# Common lambda-grid (shared by every task)",
        json.dumps(LAMBDA_GRID),
        "## Task (predict lambda_star)",
        "Pick **exactly one** lamda from the common grid above that minimises",
        "test error for this task. Output just that number—no extra text.",
        "```json",
        json.dumps(new_entry, indent=2),
        "```",
        ]
        return "\n".join(prompt_parts)


    past_entries = [
        t.metadata(include_lambda=True, task_id=f"T{i+1:02d}")
        for i, t in enumerate(context_tasks)
    ]

    new_entry = current_task.metadata(include_lambda=False, task_id="NEW")

    
    prompt_parts = [
        "You are a statistics assistant. "
        "Your task is to inspect several past Gaussian-classification problems that were solved with ridge regression "
        "and then pick the optimal ridge-regularisation constant lambda for ONE new problem (task_id: NEW). ",
        "Each task describes a two-class Gaussian problem with:",
        "  • n1, n2   : sample counts for classes 1 and 2;",
        "  • mu1, mu2 : mean vectors of the two classes;", 
        "  • alpha1, alpha2 : AR(1) Toeplitz correlation coefficients",
        "    defining each class's covariance Sigma_ij = alpha^{|i-j|}.",
        "Choose lambda only from the common grid provided below. ",
        "# Common lambda-grid (shared by every task)",
        json.dumps(LAMBDA_GRID),
        "",
        "## Past tasks with known optimal lambda",
        "```json",
        json.dumps(past_entries, indent=2),
        "```",
        "",
        "## New task (predict lambda_star)",
        "Pick **exactly one** lambda from the common grid above that minimises",
        "test error for this task. Output just that number—no extra text.",
        "```json",
        json.dumps(new_entry, indent=2),
        "```",
    ]
    return "\n".join(prompt_parts)

def llm_call(model: str, prompt: str, temperature: float = 0.0, 
             base_url: str = None, api_key: str = None) -> str:
    """
    Call the LLM API with configurable base URL and API key.
    
    Args:
        model: The model name to use
        prompt: The input prompt
        temperature: Sampling temperature
        base_url: Base URL for the API endpoint (optional)
        api_key: API key for authentication (optional) 
    """
    
    
    try:
        import openai
    except ImportError:
        raise ImportError("OpenAI package not installed. Run: pip install openai")

    # Configure client
    client_kwargs = {}
    if base_url:
        client_kwargs['base_url'] = base_url
        print(f"  Using base URL: {base_url}")
    if api_key:
        client_kwargs['api_key'] = api_key
        print(f"  Using provided API key.")    
    if not client_kwargs:
        print("  Warning: No base_url or api_key provided, using default OpenAI config")
    
    client = openai.OpenAI(**client_kwargs)
    
    print(f"  Calling {model} with temperature {temperature}")
    print(f"  Prompt length: {len(prompt)} characters")
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=5,
        stop=None,
    )
    
    result = response.choices[0].message.content.strip()
    print(f"  Response: {result}")
    return result



def run_experiment(
    n_rep: int,
    model: str,
    k_grid: List[int],
    temperature: float,
    d: int,
    log_dir: str = ".",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run experiments to evaluate model performance on lambda prediction tasks.

    Args:
        n_rep: Number of repetitions per k value
        model: Model type ("logistic-classifier", "mean", or LLM model name)
        k_grid: List of context task counts to test
        temperature: Sampling temperature for LLM calls
        d: Dimension of the tasks
        log_dir: Directory to save results (default: current directory)
        base_url: Optional base URL for the LLM API (forwarded to llm_call)
        api_key: Optional API key for the LLM API (forwarded to llm_call)

    Returns:
        DataFrame containing all experimental records
    """
    records = []

    for _ in tqdm(range(n_rep), desc="Repetitions"):
        for k in tqdm(k_grid, desc="Context sizes", leave=False):
            context_tasks = [sample_task(d) for _ in range(k)]
            current_task  = sample_task(d)

            best_lambda, _, errors = current_task.lambda_star()
            lambda_grid = LAMBDA_GRID

            lookup = dict(zip(lambda_grid, errors))
            prompt = create_prompt(current_task=current_task, context_tasks=context_tasks)

            if model == "logistic-classifier":
                if not context_tasks:
                    print("no context tasks, picking randomly")
                    lambda_pred = random.choice(lambda_grid)
                else:
                    lambdas = [t.lambda_star()[0] for t in context_tasks]
                    unique_lambdas = np.unique(lambdas)
                    if len(unique_lambdas) == 1:
                        # All tasks have the same lambda, just return that value
                        lambda_pred = unique_lambdas[0]
                    else:
                        clf = LogisticRegression(max_iter=1000)
                        X = np.array([t.vectorize() for t in context_tasks])
                        y = np.array(lambdas).astype(str)
                        clf.fit(X, y)
                        lambda_pred = float(clf.predict(current_task.vectorize().reshape(1, -1))[0])


            elif model == "mean":
                if not context_tasks:
                    print("no context tasks, picking randomly")
                    lambda_pred = random.choice(lambda_grid)
                else:
                    lambdas = [t.lambda_star()[0] for t in context_tasks]
                    lambda_pred = 10 ** np.mean(np.log10(lambdas))
                    lambda_pred = min(lookup.keys(), key=lambda x: abs(x - lambda_pred))

            else:
                # LLM-based prediction: retry up to 20 times for valid output
                for _ in range(20):
                    raw = llm_call(
                        model=model,
                        prompt=prompt,
                        temperature=temperature,
                        base_url=base_url,
                        api_key=api_key,
                    )
                    lambda_pred = extract_lambda(raw)
                    if lambda_pred is not None:
                        lambda_pred = min(lookup.keys(), key=lambda x: abs(x - lambda_pred))
                        break
                else:
                    raise RuntimeError("LLM never produced a valid lambda")

            regret    = lookup[lambda_pred] - lookup[best_lambda]
            assert regret >= 0, "Regret should be non-negative"
            idx_pred  = lambda_grid.index(lambda_pred)
            idx_best  = lambda_grid.index(best_lambda)
            grid_dist = abs(idx_pred - idx_best)

            records.append({
                "k": k,
                "lambda_pred": lambda_pred,
                "best_lambda": best_lambda,
                "regret": regret,
                "is_correct": int(grid_dist == 0),
                "grid_dist": grid_dist,
            })

    if model in baselines:
        filename = f"records_{model}_N{n_rep}.csv"
    else:
        filename = f"records_{model}_T{temperature}_N{n_rep}.csv"

    df = pd.DataFrame(records)
    local_path = os.path.join(log_dir, filename)
    os.makedirs(log_dir, exist_ok=True)
    df.to_csv(local_path, index=False)
    print(f"Saved records to {local_path}")
    return df


def parse_k_grid(s: str):
    """
    Parse k_grid from CLI.

    Accepts either:
      - Comma-separated list: "0,1,2,5,10"
      - Slice-style range:    "start:stop:step" (stop inclusive if it lands on the grid)
        e.g., "0:10:2" -> [0, 2, 4, 6, 8, 10]
    """
    s = s.strip()
    if ":" in s:
        parts = s.split(":")
        if len(parts) not in (2, 3):
            raise argparse.ArgumentTypeError(
                "Range must be 'start:stop[:step]'"
            )
        start = int(parts[0])
        stop  = int(parts[1])
        step  = int(parts[2]) if len(parts) == 3 else 1
        if step <= 0:
            raise argparse.ArgumentTypeError("Step must be a positive integer.")
        vals = list(range(start, stop + (0 if (stop - start) % step else 0), step))
        # Ensure inclusive stop if step lands exactly on it
        if vals and vals[-1] != stop and (stop - start) % step == 0:
            vals.append(stop)
        # Guard: if range overshot due to arithmetic, fix it
        vals = [k for k in vals if (start <= k <= stop) or (start >= k >= stop)]
        return vals
    # Comma list
    try:
        return [int(x.strip()) for x in s.split(",") if x.strip() != ""]
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            "k_grid must be a comma-separated list of integers or a 'start:stop[:step]' range."
        ) from e


def main():
    parser = argparse.ArgumentParser(
        prog="lambda_experiment",
        description="Run experiments to evaluate model performance on optimal lambda prediction for ridge classification task.",
        epilog=dedent(
            """\
            Examples:
              # Default grid, mean baseline, 100 reps, d=10
              python llm_exp.py --rep 100 --model mean --temperature 0.1 --dim 10

              # Custom grid as list
              python llm_exp.py --k-grid 0,1,2,5,10,20 --dim 8

              # Custom grid as range (start:stop:step)
              python llm_exp.py --k-grid 0:100:10 --rep 200 --dim 16 --log-dir ./runs

              # Use logistic classifier baseline
              python llm_exp.py --model logistic-classifier --k-grid 0,1,2,5 --dim 12

              # With custom API gateway + key
              python llm_exp.py --model qwen2.5-72b-instruct --base-url https://your.gateway/v1 --api-key sk-XXXX --dim 10
            """
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--rep", "-N", type=int, default=100,
                        help="Number of repetitions per k value (default: 100).")
    parser.add_argument("--model", type=str, default="qwen2.5-72b-instruct",
                        help='LLM model or baseline: "logistic-classifier", "mean", or an LLM name.')
    parser.add_argument("--k-grid", type=parse_k_grid,
                        default=[0, 1, 2, 5, 10, 15, 20, 50, 100], metavar="GRID",
                        help="Comma list '0,1,2' or range 'start:stop[:step]'.")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature for LLM calls (default: 0.1).")
    parser.add_argument("--dim", "-d", type=int, default=2,
                        help="Task dimensionality d (default: 2).")
    parser.add_argument("--log-dir", type=str, default=".",
                        help='Directory to save result CSVs (default: ".").')
    parser.add_argument("--base-url", type=str, default=None,
                        help="Base URL for the LLM API endpoint (e.g., an OpenAI-compatible gateway).")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key for the LLM provider. If omitted, uses environment/default client config.")

    args = parser.parse_args()
    
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")


    run_experiment(
        n_rep=args.rep,
        model=args.model,
        k_grid=args.k_grid,
        temperature=args.temperature,
        d=args.dim,
        log_dir=args.log_dir,
        base_url=args.base_url,
        api_key=api_key,
    )


if __name__ == "__main__":
    main()
 

