import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    names = [
        "Token Count",
        "Node Count",
        "Halstead Difficulty",
        "Cyclomatic Complexity",
    ]
    tokens, nodes, halstead, cyclomatic = [], [], [], []
    for file in Path("../../../braincode/.cache/profiler").glob("en*.benchmark"):
        with open(file, "r") as f:
            data = json.loads(f.read())
            tokens.append(data["token_counts"])
            nodes.append(data["ast_node_counts"])
            halstead.append(data["program_difficulty"])
            cyclomatic.append(data["cyclomatic_complexity"])
    static_properties = np.array([tokens, nodes, halstead, cyclomatic])
    static_corrs = np.corrcoef(static_properties)
    df = pd.DataFrame(static_corrs, columns=names, index=names)
    latex = df.to_latex().replace("{lrrrr}", "{l|rrrr}")
    with open(f"static_property_corrs.tex", "w") as f:
        f.write(latex)


if __name__ == "__main__":
    main()
