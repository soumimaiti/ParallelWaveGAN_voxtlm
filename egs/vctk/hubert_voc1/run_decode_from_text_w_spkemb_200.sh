#!/bin/bash

. ./path.sh


#Path to data - contains both tokens and spk emb
dataroot="test_data/test_200" 

xvector_dir="${dataroot}/test-clean/"

# create utt2xvector from xvector.ark and xvector.scp
python local/data_prep_spemb.py --ark "${xvector_dir}/xvector.ark" --scp "${xvector_dir}/xvector.scp" --utt2spkemb "${dataroot}/utt2xvector"

# HiFi-GAN model
# model checkpoint number
steps=300000
# model checkpoint path
expdir="./exp/tr_no_dev_vctk_hifigan_hubert_large_km200_24khz.v1"
checkpoint="${expdir}/checkpoint-${steps}steps.pkl"



nutt=$(<"${dataroot}/hyp_tok" wc -l)

for i in $(seq 1 200 ${nutt}); do
    python local/decode_from_text_w_spkemb.py \
        --text "${dataroot}/hyp_tok" \
        --utt2spkemb "${dataroot}/utt2xvector" \
        --outdir "${dataroot}/waves_step${steps}" \
        --checkpoint ${checkpoint} \
        --num_utts 100 \
        --start ${i}|| exit 1;
done
