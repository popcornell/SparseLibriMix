import argparse
import os
import random
import numpy as np
import json
import glob
import soundfile as sf
from pathlib import Path
import collections


parser = argparse.ArgumentParser("Generating mixtures")
parser.add_argument("json_file")
parser.add_argument("noise_dir")
parser.add_argument("out_json")
parser.add_argument("--n_mixtures", default=1000,  type=int)
parser.add_argument("--n_speakers", default=3,  type=int)
parser.add_argument('--ovr_ratio', type=float, default=0.0,
                    help='target overlap amount')
parser.add_argument("--maxlength", type=int, default=15)
parser.add_argument('--random_seed', type=int, default=777,
                    help='random seed')
parser.add_argument("--version", type=int, default=1)


def find_sub_utts_subsets(c_utt, minlen):
    valid = []
    for i in range(len(c_utt)): # O(n**2)
        sum_till_now = 0
        for j in range(i+1, len(c_utt)):
            sum_till_now += c_utt[j]["stop"] - c_utt[j]["start"]
            if sum_till_now <= minlen:
                valid.append([sum_till_now, [i, j]])

    # select a random utterance from the valid ones
    # selection can be conditioned on the contiguous subarray which goes nearest to minlen
    if valid:
        valid = sorted(valid, key= lambda x : abs(x[0]-minlen))[0]
        start, stop = valid[-1]
        return c_utt[start:stop+1]
    else:
        return [np.random.choice(c_utt)] # all utterances are longer


if __name__ == "__main__":
    args = parser.parse_args()
    VERSION = args.version

    os.makedirs(Path(args.out_json).parent, exist_ok=True)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    f_name = Path(args.noise_dir).name + f'_max{args.maxlength}s.json'
    if os.path.isfile(f_name):
        print("Loading noise file lists")
        with open(f_name, 'r') as f:
            noises = json.load(f)
    else:
        noises = glob.glob(os.path.join(args.noise_dir, "**/*.wav"), recursive=True)
        prev_len = len(noises)
        noises = [x for x in noises if len(sf.SoundFile(x)) >= args.maxlength*sf.SoundFile(x).samplerate]
        print("Number of noise wavs : {}".format(len(noises)))
        print("Discarded : {}".format(prev_len - len(noises)))
        with open(f_name, 'w') as f:
            json.dump(noises, f)
    ######

    with open(args.json_file, "r") as f:
        utterances = json.load(f)

    all_speakers = list(utterances.keys())
    total_metadata = []
    mix_n = 0
    while mix_n < args.n_mixtures:

        # recording ids are mix_0000001, mix_0000002, ...
        recid = 'mix_{:07d}'.format(mix_n + 1)
        c_speakers = random.sample(all_speakers, args.n_speakers) # could be made weighted
        metadata = []
        maxlength = -1

        # we sample all utterances at once from different librispeech files
        utts = [random.choice(utterances[spk]) for spk in c_speakers]
        # we get min length in seconds for all speakers

        mindur_spk = np.inf
        for spk_indx in range(len(utts)):
            tmp = 0
            for sub_utt in utts[spk_indx]:
                tmp += sub_utt["stop"] - sub_utt["start"]
            mindur_spk = min(tmp, mindur_spk)

        # having minimum duration we keep adding utterances from one speaker till we have minimum duration
        kept = []
        for spk_indx in range(len(utts)):
            tmp = find_sub_utts_subsets(utts[spk_indx], mindur_spk)
            kept.append(tmp[::-1]) # we use pop after thus we reverse here
        utts = kept

        lasts = {}
        for i in c_speakers:
            lasts[i] = [0, 0]

        overlap_stat = 0
        tot = 0
        sub_utt_num = 0
        while np.any(utts): # till we have utterances
            if tot == 0:
                spk_indx = 0
                prev_spk_indx = 1
            else:
                prev_spk_indx = spk_indx
                spk_indx = random.choice([x for x in range(len(c_speakers)) if x != prev_spk_indx])

            # if number of sub_utts for this speaker is greater than n sub utts of all other
            # we can afford to not overlap this utterance on the left
            try:
                sub_utt = utts[spk_indx].pop()
            except:
                continue # no more utterances for this speaker

            c_spk = c_speakers[spk_indx]
            prev_spk = c_speakers[prev_spk_indx]

            if lasts[prev_spk][-1] != 0:
                # not first utterance
                if VERSION == 1:
                    stop = min([x for x in [lasts[x][-1] for x in c_speakers if x != c_spk] if x != 0])
                    start = max([x for x in [lasts[x][0] for x in c_speakers if x != c_spk] if x != 0])
                    lastlen = stop - start
                    if args.ovr_ratio != 0:
                        raise NotImplemented
                    else:
                        # This should always be the stop of the previous speaker.
                        it = max([x for x in [lasts[x][-1] for x in c_speakers] if x != 0]) + 0.05
                elif VERSION == 2:
                    start, stop = lasts[prev_spk]
                    lastlen = stop - start
                    if args.ovr_ratio != 0:
                        raise NotImplemented
                    else:
                        # This should always be the stop of the previous speaker.
                        it = max([x for x in [lasts[x][-1] for x in c_speakers] if x != 0]) + 0.05
                elif VERSION == 3:
                    c_length = sub_utt["stop"] - sub_utt["start"]
                    stop = min([x for x in [lasts[x][-1] for x in c_speakers if x != c_spk] if x != 0])
                    start = max([x for x in [lasts[x][0] for x in c_speakers if x != c_spk] if x != 0])
                    lastlen = stop - start
                    if args.ovr_ratio != 0:
                        raise NotImplemented

                    else:
                        # This should always be the stop of the previous speaker.
                        it = max([x for x in [lasts[x][-1] for x in c_speakers] if x != 0]) + 0.05
                else:
                    raise ValueError

                # if offset == 0# no overlap maybe we can use pauses between utterances
            else:
                it = np.random.uniform(0.2, 0.5)  # first utterance

            maxlength = max(maxlength, it + (sub_utt["stop"] - sub_utt["start"]))

            c_meta = {"file": "/".join(sub_utt["file"].split("/")[-3:]), "words": sub_utt["words"],
                      "spk_id": sub_utt["spk_id"],
                      "chapter_id": sub_utt["chapter_id"], "utt_id": sub_utt["utt_id"],
                      "start": np.round(it, 3),
                      "stop": np.round(it + (sub_utt["stop"] - sub_utt["start"]), 3),
                      "orig_start": sub_utt["start"],
                      "orig_stop": sub_utt["stop"], "lvl":  np.random.uniform( -33, -25),
                      "source": "s{}".format(spk_indx + 1), "sub_utt_num": sub_utt_num }
            metadata.append(c_meta)
            lasts[c_spk][0] = c_meta["start"]
            lasts[c_spk][1] = c_meta["stop"]  # can't overlap with itself
            tot += c_meta["stop"] - c_meta["start"]
            sub_utt_num += 1

        ## noise ##
        maxlength += np.random.uniform(0.2, 0.5) # ASR purposes we add some silence at end
        noise = np.random.choice(noises)
        # if noisefile is more than maxlength then we take a random window
        if len(sf.SoundFile(noise)) - int(maxlength * sf.SoundFile(noise).samplerate) <= 0 :
            print("TEST ONLY too long utterance skipping utt")
            continue
        offset = random.randint(0, len(sf.SoundFile(noise)) - int(maxlength*sf.SoundFile(noise).samplerate))
        c_lvl = np.random.uniform(-38, -30) #np.clip(first_lvl - random.normalvariate(3.47, 4), -40, 0)

        metadata.append({"file": noise.split("/")[-1],
                         "start": 0,
                         "stop": maxlength, "orig_start": np.round(offset/sf.SoundFile(noise).samplerate, 3),
                         "orig_stop": np.round(offset/sf.SoundFile(noise).samplerate + maxlength, 3),
                         "lvl": c_lvl, "source": "noise", "channel": random.randint(0,1)})

        mixture_metadata = {"mixture_name": recid}
        for elem in metadata:
            if elem["source"] not in mixture_metadata.keys():
                mixture_metadata[elem["source"]] = [elem]
            else:
                mixture_metadata[elem["source"]].append(elem)

        total_metadata.append(mixture_metadata)
        mix_n += 1

    with open(os.path.join(args.out_json), "w") as f:
        json.dump(total_metadata, f, indent=4)


























