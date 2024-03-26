import typer
import json
from pathlib import Path


def createjson(
    dataset_path: Path = typer.Option(...),
    split: str = typer.Option(...),
    output_json_file_path: Path = typer.Option(...),
):
    data = {}
    data["version"] = 1
    data["date"] = "2023.08.31"
    data["videos"] = []

    split_dataset_path = dataset_path / split
    video_paths = list(split_dataset_path.rglob("*.mkv"))
    for video_path in video_paths:
        video_features_path = (
            video_path.parent / f"{video_path.stem}_ResNET_TF2_PCA512.npy"
        )

        video = {}
        video["path_video"] = video_path.as_posix()
        video["path_features"] = video_features_path.as_posix()

        video_labels_path = video_path.parent / "Labels-v2.json"
        with open(video_labels_path) as f:
            label = json.load(f)

        video["annotations"] = [
            annotation
            for annotation in label["annotations"]
            if int(annotation["gameTime"][0]) == int(video_path.stem[0])
        ]
        for annotation in video["annotations"]:
            annotation["position"] = int(annotation["position"])

        data["videos"].append(video)

    data["labels"] = [
        "Penalty",
        "Kick-off",
        "Goal",
        "Substitution",
        "Offside",
        "Shots on target",
        "Shots off target",
        "Clearance",
        "Ball out of play",
        "Throw-in",
        "Foul",
        "Indirect free-kick",
        "Direct free-kick",
        "Corner",
        "Yellow card",
        "Red card",
        "Yellow->red card",
    ]

    with open(output_json_file_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Json file saved at: {output_json_file_path}")


if __name__ == "__main__":
    typer.run(createjson)
