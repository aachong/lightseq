#!/usr/bin/env bash

export http_proxy=http://bj-rd-proxy.byted.org:3128 
export https_proxy=http://bj-rd-proxy.byted.org:3128

set -ex
THIS_DIR=$(dirname $(readlink -f $0))
cd $THIS_DIR/../../..

pip3 install -e ./

until [[ -z "$1" ]]
do
    case $1 in
        --pretrain_dir)
            shift; PRETRAIN_DIR=$1;
            shift;;
        --save_hdfs_dir)
            shift; SAVE_HDFS_DIR=$1;
            shift;;
        --encoder_int4)
            shift; ENCODER_INT4=$1;
            shift;;
        --decoder_int4)
            shift; DECODER_INT4=$1;
            shift;;
        *)
            shift;;
    esac
done

# max_epoch默认为60
MAX_EPOCH=50
LR=5e-5

hdfs dfs -get $PRETRAIN_DIR /tmp/checkpoint.pt

if [ ! -d "/tmp/wmt14_en_de" ]; then
    echo "Downloading dataset"
    wget http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/lightseq/wmt_data/databin_wmt14_en_de.tar.gz -P /tmp
    tar -zxvf /tmp/databin_wmt14_en_de.tar.gz -C /tmp && rm /tmp/databin_wmt14_en_de.tar.gz
fi

lightseq-train /tmp/wmt14_en_de/ \
    --task translation \
    --save-dir /tmp/save_dir \
    --arch ls_transformer_wmt_en_de --share-decoder-input-output-embed \
    --optimizer ls_adam --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr $LR --lr-scheduler inverse_sqrt --warmup-updates 4000 --weight-decay 0.0001 \
    --criterion ls_label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --fp16 --max-epoch $MAX_EPOCH \
    --use-torch-layer \
    --enable-quant \
    --quant-mode qat \
    --quant-num-bits 8 \
    --int4-encoder-layer $ENCODER_INT4 --int4-decoder-layer $DECODER_INT4 \
    --finetune-from-model /tmp/checkpoint.pt --keep-last-epochs 5

if [[ ${SAVE_HDFS_DIR} ]]; then
    hdfs dfs -mkdir -p $SAVE_HDFS_DIR
    hdfs dfs -put /tmp/save_dir/* $SAVE_HDFS_DIR
fi
