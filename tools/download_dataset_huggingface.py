from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(
        description="Download Datasets from HuggingFace.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
parser.add_argument("--dataset_repo", type=str, default=None, help="repo id of the dataset")
parser.add_argument("--output_dir", type=str, default=None, help="path to save the dataset")
parser.add_argument("--allow_patterns", type=str, default="*", help="filter files to download")

args = parser.parse_args()

from huggingface_hub import snapshot_download

snapshot_download(repo_id=args.dataset_repo,
                  repo_type="dataset", revision="main",
                  local_dir=args.output_dir,
                  allow_patterns=args.allow_patterns)