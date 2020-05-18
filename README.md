# SparseLibriMix

### About the dataset
This repo contains the sparse version of [LibriMix](), SparseLibriMix (SLMix),  an open source dataset for source separation in noisy 
environments and with variable overlap-ratio. Both are derived from LibriSpeech (clean subset) 
and WHAM noise. Due to insufficient noise material this is a test-set-only version. It was created 
in order to evaluate supervised source separation algorithms in more realistic scenarios where speakers do not overlap 100 % of the time as it happens in most currently used datasets.  

### Generating SLMix
To generate SLMix, clone the repo, install the requirements and run the main script : 
[`create_sparse.sh`](./create_sparse.sh)

NOTE: modify the script with you paths first and download LibriSpeech test-clean and WHAM noises. 

```
git clone https://github.com/popcornell/SparseLibriMix
cd SparseLibriMix 
pip install -r requirements.txt
./create_sparse.sh
```
  
By default, it will generate 500 utterances for noisy and clean mixtures, for 
6 different overlap ratios, at 8 kHz and 16 kHz and for 3 and 2 speakers. 
The total number of utterances is then 500 * 2 * 6 * 2 * 2 =  24000. 

---
####Note
We provide directly the metadata for the purpose of generating the "official" test-set only dataset.  
However we also provide metadata generation scripts, which can be used to generate an arbitrarily amount of data at least 
in clean setting (as WHAM noises of sufficient length are unavailable for a properly sized training set). 


We plan to extend this in future to have a training set and a validation set derived from clean subset of LibriSpeech. 

