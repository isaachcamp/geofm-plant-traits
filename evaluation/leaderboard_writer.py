
import json
import os
from datetime import datetime
from pathlib import Path

def update_leaderboard(model_name, model_path, results):
    """Update the leaderboard with new results."""
    leaderboard_path = Path("leaderboard/leaderboard.json")

    # Create directory if it doesn't exist
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing leaderboard or create a new one
    if os.path.exists(leaderboard_path):
        with open(leaderboard_path, "r") as f:
            leaderboard = json.load(f)
    else:
        leaderboard = []

    # Get git info if available
    commit_hash = os.environ.get("GITHUB_SHA", "local")
    author = os.environ.get("GITHUB_ACTOR", "local")

    # Create entry
    entry = {
        "model_name": model_name,
        "model_path": str(model_path),
        "results": results,
        "timestamp": datetime.now().isoformat(),
        "commit": commit_hash,
        "author": author
    }

    # Add or update entry
    found = False
    for i, item in enumerate(leaderboard):
        if item["model_name"] == model_name:
            leaderboard[i] = entry
            found = True
            break

    if not found:
        leaderboard.append(entry)

    # Sort by R2 score
    leaderboard = sorted(leaderboard, key=lambda x: x["results"]["mean_r2_score"], reverse=True)

    # Save updated leaderboard
    with open(leaderboard_path, "w") as f:
        json.dump(leaderboard, f, indent=2)

    # Also create a markdown version for better GitHub rendering
    create_markdown_leaderboard(leaderboard)

    return leaderboard

def create_markdown_leaderboard(leaderboard):
    """Create a markdown version of the leaderboard."""
    md_path = Path("leaderboard/README.md")

    with open(md_path, "w") as f:
        f.write("# Machine Learning Models Leaderboard\n\n")
        f.write("Last updated: {}\n\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        f.write("## Overall Rankings\n\n")
        f.write("| Rank | Model | Author | Mean $R^2$ Score | Last Updated |\n")
        f.write("|------|-------|--------|--------------|-------------|\n")

        for i, entry in enumerate(leaderboard):
            f.write("| {} | {} | {} | {:.4f} | {} |\n".format(
                i + 1,
                entry["model_name"],
                entry["author"],
                entry["results"].pop("mean_r2_score"),
                entry["timestamp"].split("T")[0]
            ))

        f.write("\n## Detailed Results\n\n")

        for var in leaderboard[0]["results"].keys():
            f.write(f"### {var.upper()}\n\n")
            f.write("| Rank | Model | $R^2$ Score | RMSE | MAE | MAPE |\n")
            f.write("|------|-------|----------|----------|-----------|--------|\n")

            # Sort models by their performance on this variable
            sorted_entries = sorted(
                [e for e in leaderboard if var in e["results"]],
                key=lambda x: x["results"][var]["R_squared"],
                reverse=True
            )

            for i, entry in enumerate(sorted_entries):
                results = entry["results"].get(var, {})
                if results:
                    f.write("| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |\n".format(
                        i + 1,
                        entry["model_name"],
                        results.get("R_squared", 0),
                        results.get("RMSE", 0),
                        results.get("MAE", 0),
                        results.get("MAPE", 0)
                    ))
            f.write("\n")
