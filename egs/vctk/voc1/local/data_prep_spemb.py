import kaldiio
import numpy as np
import os
import argparse
import tqdm

parser = argparse.ArgumentParser(
    description=""
)
parser.add_argument(
    "--ark",
    required=True,
    type=str,
    help="kaldi-style ark file.",
)
parser.add_argument(
    "--scp",
    required=True,
    type=str,
    help="kaldi-style scp file.",
)

parser.add_argument(
    "--utt2spkemb",
    default=None,
    type=str,
    help="utt2spkemb xvector file.",
)
args = parser.parse_args()

# Read utt_id lis from scp
utt_id_list=[]
with open(args.scp) as f:
    for line in f:
        key=line.split()[0]
        utt_id_list.append(key)
fw = open(args.utt2spkemb, "w")
with kaldiio.ReadHelper('ark:{}'.format(args.ark)) as reader:
    for key, numpy_array in tqdm.tqdm(reader):
        if key in utt_id_list:
            fw.write("{} {}\n".format(key, " ".join(map(str, numpy_array))))
fw.close()
