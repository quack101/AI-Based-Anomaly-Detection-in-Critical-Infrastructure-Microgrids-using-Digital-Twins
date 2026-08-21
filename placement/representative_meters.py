import json
from pathlib import Path


def save_representative_meters(selected, output_file):
    """
    Save the selected representative meters to a JSON file.
    """

    output_file = Path(output_file)

    # Create the simulation directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    meters = []

    for i, load in enumerate(selected, start=1):

        meters.append({

            "meter_id": f"M{i:03d}",

            "load_name": load["name"],

            "bus": load["bus"],

            "base_kw": load["base_kw"],

            "branch": load["branch"],

            "depth": load["depth"],

            "downstream_kw": load["downstream_kw"],

            "size_class": load["size_class"],

            "on_backbone": load["on_backbone"]
        })

    with open(output_file, "w") as f:
        json.dump(meters, f, indent=4)

    print(f"\nSaved {len(meters)} representative meters")
    print(f"Output: {output_file}")