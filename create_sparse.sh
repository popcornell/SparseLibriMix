#!/bin/bash

librispeech_subdir=/media/sam/Data/LibriSpeech/test-clean/ # path to LibriSpeech test-clean
noise_dir=/media/sam/Data/WSJ/wham_noise/tt # path to test WHAM noises
metadata_dir=./metadata
out_dir=/home/sam/Desktop/temp/to_listen/sparse_libri_def # output directory
stage=0
fs=16000
all_overlap="0.2 0.4 0.6 0.8 1"

set -e
mkdir -p $out_dir

if [[ $stage -le 0 ]]; then
    for fs in 8000 16000; do
      for n_speakers in 2 3; do
        for ovr_ratio in 0 $all_overlap; do
          echo "Making mixtures for ${n_speakers} speakers and overlap ${ovr_ratio}"
          python scripts/make_mixtures.py $metadata_dir/sparse_${n_speakers}_${ovr_ratio}/metadata.json \
            $librispeech_subdir $out_dir/sparse_${n_speakers}_${ovr_ratio}/wav${fs} --noise_dir $noise_dir --rate $fs
          done
      done
    done
fi