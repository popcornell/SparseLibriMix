import os
import glob
from pathlib import Path
from .textgrid_utils import build_hashtable_textgrid, get_textgrid_sa
import soundfile as sf
import numpy as np


def hash_librispeech(librispeech_traintest):

    hashtab = {}
    utterances = glob.glob(os.path.join(librispeech_traintest, "**/*.wav"), recursive=True)
    for utt in utterances:
        id = Path(utt).parent.parent
        hashtab[id] = utt
    return hashtab

def ema_energy(x, alpha=0.99):

    out = np.sum(x[0]**2)
    for i in range(1, len(x)):
        out = (1-alpha)*np.sum(x[i]**2) + alpha*out
    return out

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def estimate_snr(speech, file):

    def get_energy(signal): # probably windowed estimation is better but we take min afterwards and
        return np.sum(signal**2)

    audio, fs = sf.read(file)
    audio = audio - np.mean(audio)
    s, e = [int(x*fs) for x in speech[0]]
    speech_lvl = [get_energy(audio[s:e])]
    noise_lvl = [get_energy(audio[0: s])]

    for i in range(1, len(speech)):
        speech_lvl.append(get_energy(audio[int(speech[i][0]*fs):int(speech[i][-1]*fs)]))
        noise_lvl.append(get_energy(audio[int(speech[i-1][-1]*fs): int(speech[i][0]*fs)]))

    noise_lvl.append(get_energy(audio[int(speech[-1][-1]*fs):]))

    noise_lvl = min(noise_lvl) # we take min to avoid breathing
    speech_lvl = min(speech_lvl)

    return speech_lvl / (noise_lvl + 1e-8)


def build_utterance_list(librispeech_dir, textgrid_dir, merge_shorter=0.15, fs=16000):

    hashgrid = build_hashtable_textgrid(textgrid_dir)
    audiofiles = glob.glob(os.path.join(librispeech_dir, "**/*.flac"), recursive=True)

    utterances = {}
    tot_missing = 0

    snrs = []

    for f in audiofiles:

        filename = Path(f).stem
        if filename not in hashgrid.keys():
            print("Missing Alignment file for : {}".format(f))
            tot_missing += 1
            continue
        speech, words = get_textgrid_sa(hashgrid[filename], merge_shorter)

        spk_id = Path(f).parent.parent.stem

        sub_utterances = []
        # get all segments for this speaker
        if not speech:
            raise EnvironmentError("something is wrong with alignments or parsing, all librispeech files have speech")

        snr = estimate_snr(speech, f)
        snrs.append([f, snr])

        for i in range(len(speech)):
            start, stop = speech[i]
            #start = #int(start*fs)
            #stop = int(stop*fs)
            tmp = {"textgrid": hashgrid[filename], "file": f, "start": start, "stop": stop, "words": words[i],
                   "spk_id": spk_id, "chapter_id": Path(f).parent.stem, "utt_id": Path(f).stem}
            sub_utterances.append(tmp)

        if spk_id not in utterances.keys():
            utterances[spk_id] = [sub_utterances]
        else:
            utterances[spk_id].append(sub_utterances)

    # here we filter utterances based on SNR we sort the snrs list and if a chapter has more than 10 entries with snr lower than
    # 10 we remove that chapter
    snrs = sorted(snrs, key = lambda x : x[-1])

    chapters = {}
    for x in snrs:
        file, snr = x
        chapter = Path(file).parent.stem
        if chapter not in chapters.keys():
            chapters[chapter] = [0, 0]
        chapters[chapter][0]  += 1
        if snr < 25: # tuned manually
            chapters[chapter][-1] += 1

    # normalize
    for k in chapters.keys():
        chapters[k] = chapters[k][-1] / chapters[k][0]

    prev_tot_utterances = sum([len(utterances[k]) for k in utterances.keys()])
    new = {}
    for spk in utterances.keys():
        new[spk] = []
        for utt in utterances[spk]:
            if chapters[utt[0]["chapter_id"]] >= 0.1:
                continue
            else:
                new[spk].append(utt)
        if len(new[spk]) == 0:
            continue

    utterances = new


    print("Discarded {} over {} files because of low SNR".format(prev_tot_utterances - \
                                                               sum([len(utterances[k]) for k in utterances.keys()]), prev_tot_utterances))

    return utterances

if __name__ == "__main__":
    build_utterance_list("/media/sam/Data/LibriSpeech/test-clean/", "/home/sam/Downloads/librispeech_alignments/test-clean/")
