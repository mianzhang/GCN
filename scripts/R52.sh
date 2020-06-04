dataset=R52

corpus_dir=data/${dataset}/corpus
dataset_dir=data/${dataset}/dataset
ckpt_dir=data/${dataset}/ckpt

do_what=$1

if [ "${do_what}" == "clean" ]; then
    python -u clean.py --dataset=${dataset} > log/clean.${dataset}

elif [ "${do_what}" == "build_dataset" ]; then
    python -u build_dataset.py --dataset=${dataset} > log/build_dataset.${dataset}

elif [ "${do_what}" == "build_graph" ]; then
    python -u build_graph.py --train=${dataset_dir}/train.txt \
        --dev=${dataset_dir}/dev.txt \
        --test=${dataset_dir}/test.txt \
        --data=${ckpt_dir}/data.pkl > log/build_graph.${dataset}

elif [ "${do_what}" == "train" ]; then
    python -u train.py --data=${ckpt_dir}/data.pkl \
        --model=save/model.${dataset}.pt \
        --from_begin --device=cuda:3 --epochs=200 > log/train.${dataset}
fi
