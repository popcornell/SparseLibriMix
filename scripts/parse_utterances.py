import argparse
from utils.librispeech_utils import build_utterance_list
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser("Parsing Librispeech Utterances to json file")
parser.add_argument("librispeech_dir")
parser.add_argument("textgrid_dir")
parser.add_argument("out_file")
parser.add_argument("--merge_shorter", type=float, default=0.15)



if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(Path(args.out_file).parent, exist_ok=True)
    utterances = build_utterance_list(args.librispeech_dir, args.textgrid_dir, merge_shorter=args.merge_shorter)

    with open(args.out_file, "w") as f:
        json.dump(utterances, f, indent=4)
