import json
import os

def save_result_to_json(result: dict, filename: str, save_dir: str = "./results"):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    with open(filepath, "w") as f:
        json.dump(result, f, indent=4)

    print(f"Zapisano wynik do: {filepath}")
