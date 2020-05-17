import argparse
import json
from tqdm import tqdm
import soundfile as sf
import numpy as np
import os
import pyloudnorm
from scipy.signal import resample_poly

parser = argparse.ArgumentParser()
parser.add_argument("json")
parser.add_argument("librispeech_dir")
parser.add_argument('out_dir',help='output data dir of mixture')
parser.add_argument("--noise_dir", type=str, default="")
parser.add_argument('--rate', type=int, default=16000,
                    help='sampling rate')

args = parser.parse_args()

if not args.noise_dir:
    print("Generating only clean version")

with open(args.json, "r") as f:
    total_meta = json.load(f)


def resample_and_norm(signal, orig, target, lvl):

    if orig != target:
        signal = resample_poly(signal, target, orig)

    #fx = (AudioEffectsChain().custom("norm {}".format(lvl)))
    #signal = fx(signal)

    meter = pyloudnorm.Meter(target, block_size=0.1)
    loudness = meter.integrated_loudness(signal)
    signal = pyloudnorm.normalize.loudness(signal, loudness, lvl)

    return signal

for mix in tqdm(total_meta):
    filename = mix["mixture_name"]
    sources_list = [x for x in mix.keys() if x != "mixture_name"]

    sources = {}
    maxlength = 0
    for source in sources_list:
        # read file optional resample it
        source_utts = []
        for utt in mix[source]:
            if utt["source"] != "noise": # speech file
                utt["file"] = os.path.join(args.librispeech_dir, utt["file"])
            else:
                if args.noise_dir:
                    utt["file"] = os.path.join(args.noise_dir, utt["file"])
                else:
                    continue

            utt_fs = sf.SoundFile(utt["file"]).samplerate
            audio, fs = sf.read(utt["file"], start=int(utt["orig_start"]*utt_fs),
                            stop=int(utt["orig_stop"]*utt_fs))

            #assert len(audio.shape) == 1, "we currently not support multichannel"
            if len(audio.shape) > 1:
                audio = audio[:, utt["channel"]] #TODO
            audio = audio - np.mean(audio) # zero mean cos librispeech is messed up sometimes
            audio = resample_and_norm(audio, fs, args.rate, utt["lvl"])
            audio = np.pad(audio, (int(utt["start"]*args.rate), 0), "constant") # pad the beginning
            source_utts.append(audio)
            maxlength = max(len(audio), maxlength)

        sources[source] = source_utts

    # pad everything to same length
    for s in sources.keys():
        for i in range(len(sources[s])):
            tmp = sources[s][i]
            sources[s][i] = np.pad(tmp,  (0, maxlength-len(tmp)), 'constant')

    # mix n sum
    tot_mixture = None
    for indx, s in enumerate(sources.keys()):
        if s == "noise":
            continue
        source_mix = np.sum(sources[s], 0)
        os.makedirs(os.path.join(args.out_dir, s), exist_ok=True)
        sf.write(os.path.join(args.out_dir, s, filename + ".wav"), source_mix, args.rate)
        if indx == 0:
            tot_mixture = source_mix
        else:
            tot_mixture += source_mix

    os.makedirs(os.path.join(args.out_dir, "mix_clean"), exist_ok=True)
    sf.write(os.path.join(args.out_dir, "mix_clean", filename + ".wav"), tot_mixture, args.rate)

    if args.noise_dir:
        s = "noise"
        source_mix = np.sum(sources[s], 0)
        os.makedirs(os.path.join(args.out_dir, s), exist_ok=True)
        sf.write(os.path.join(args.out_dir, s, filename + ".wav"), source_mix, args.rate)
        tot_mixture += source_mix
        os.makedirs(os.path.join(args.out_dir, "mix_noisy"), exist_ok=True)
        sf.write(os.path.join(args.out_dir, "mix_noisy", filename + ".wav"), tot_mixture, args.rate)























