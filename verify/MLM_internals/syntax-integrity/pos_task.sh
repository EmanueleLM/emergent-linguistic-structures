# DATASET='ted'
# MODEL='bert'
# LAYER='-9'
# ACTIVATION='linear'
# for SEED in 1
# do
#     echo "nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --activation $ACTIVATION"
#     nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --activation $ACTIVATION
#     wait && echo "DONE"
# done

# DATASET='en-universal'
# for SEED in 1
# do
#     echo "nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --activation $ACTIVATION"
#     nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --activation $ACTIVATION
#     wait && echo "DONE"
# done

# DATASET='ud-english-pud'
# for SEED in 1
# do
#     echo "nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --activation $ACTIVATION"
#     nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --activation $ACTIVATION
#     wait && echo "DONE"
# done

# DATASET='ud-english-lines'
# for SEED in 1
# do
#     echo "nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --activation $ACTIVATION"
#     nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --activation $ACTIVATION
#     wait && echo "DONE"
# done

# DATASET='ud-english-ewt'
# for SEED in 1
# do
#     echo "nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --activation $ACTIVATION"
#     nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --activation $ACTIVATION
#     wait && echo "DONE"
# done

# DATASET='ud-english-gum'
# for SEED in 1
# do
#     echo "nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --activation $ACTIVATION"
#     nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --activation $ACTIVATION
#     wait && echo "DONE"
# done
################################################
################################################
################################################
DATASET='ted'
MODEL='roberta'
ACTIVATION='linear'
for SEED in 1
do
    echo "nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --activation $ACTIVATION"
    nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --activation $ACTIVATION
    wait && echo "DONE"
done

DATASET='en-universal'
for SEED in 1
do
    echo "nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --activation $ACTIVATION"
    nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --activation $ACTIVATION
    wait && echo "DONE"
done

DATASET='ud-english-pud'
for SEED in 1
do
    echo "nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --activation $ACTIVATION"
    nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --activation $ACTIVATION
    wait && echo "DONE"
done

DATASET='ud-english-lines'
for SEED in 1
do
    echo "nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --activation $ACTIVATION"
    nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --activation $ACTIVATION
    wait && echo "DONE"
done

DATASET='ud-english-ewt'
for SEED in 1
do
    echo "nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --activation $ACTIVATION"
    nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --activation $ACTIVATION
    wait && echo "DONE"
done

DATASET='ud-english-gum'
for SEED in 1
do
    echo "nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --activation $ACTIVATION"
    nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --activation $ACTIVATION
    wait && echo "DONE"
done