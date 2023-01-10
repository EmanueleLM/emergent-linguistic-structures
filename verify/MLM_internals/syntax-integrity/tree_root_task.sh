MODEL='bert'
ACTIVATION='linear'
LAYER='-9'
DATASET='ted'
for SEED in 1
do
    echo "nice python3 tree_root_task.py --llm '$MODEL' --dataset '$DATASET' --layer $LAYER --seed $SEED --activation $ACTIVATION"
    nice python3 tree_root_task.py --llm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --activation $ACTIVATION
    wait && echo "DONE"
done

DATASET='en-universal'
for SEED in 1
do
    echo "nice python3 tree_root_task.py --llm '$MODEL' --dataset '$DATASET' --layer $LAYER --seed $SEED --activation $ACTIVATION"
    nice python3 tree_root_task.py --llm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --activation $ACTIVATION
    wait && echo "DONE"
done

DATASET='ud-english-pud'
for SEED in 1
do
    echo "nice python3 tree_root_task.py --llm '$MODEL' --dataset '$DATASET' --layer $LAYER --seed $SEED --activation $ACTIVATION"
    nice python3 tree_root_task.py --llm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --activation $ACTIVATION
    wait && echo "DONE"
done

DATASET='ud-english-lines'
for SEED in 1
do
    echo "nice python3 tree_root_task.py --llm '$MODEL' --dataset '$DATASET' --layer $LAYER --seed $SEED --activation $ACTIVATION"
    nice python3 tree_root_task.py --llm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --activation $ACTIVATION
    wait && echo "DONE"
done

DATASET='ud-english-ewt'
for SEED in 1
do
    echo "nice python3 tree_root_task.py --llm '$MODEL' --dataset '$DATASET' --layer $LAYER --seed $SEED --activation $ACTIVATION"
    nice python3 tree_root_task.py --llm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --activation $ACTIVATION
    wait && echo "DONE"
done

DATASET='ud-english-gum'
for SEED in 1
do
    echo "nice python3 tree_root_task.py --llm '$MODEL' --dataset '$DATASET' --layer $LAYER --seed $SEED --activation $ACTIVATION"
    nice python3 tree_root_task.py --llm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --activation $ACTIVATION
    wait && echo "DONE"
done

###############################################################
###############################################################
###############################################################

DATASET='ted'
MODEL='roberta'
for SEED in 1
do
    echo "nice python3 tree_root_task.py --llm '$MODEL' --dataset '$DATASET' --layer $LAYER --seed $SEED --activation $ACTIVATION"
    nice python3 tree_root_task.py --llm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --activation $ACTIVATION
    wait && echo "DONE"
done

DATASET='en-universal'
for SEED in 1
do
    echo "nice python3 tree_root_task.py --llm '$MODEL' --dataset '$DATASET' --layer $LAYER --seed $SEED --activation $ACTIVATION"
    nice python3 tree_root_task.py --llm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --activation $ACTIVATION
    wait && echo "DONE"
done

DATASET='ud-english-pud'
for SEED in 1
do
    echo "nice python3 tree_root_task.py --llm '$MODEL' --dataset '$DATASET' --layer $LAYER --seed $SEED --activation $ACTIVATION"
    nice python3 tree_root_task.py --llm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --activation $ACTIVATION
    wait && echo "DONE"
done

DATASET='ud-english-lines'
for SEED in 1
do
    echo "nice python3 tree_root_task.py --llm '$MODEL' --dataset '$DATASET' --layer $LAYER --seed $SEED --activation $ACTIVATION"
    nice python3 tree_root_task.py --llm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --activation $ACTIVATION
    wait && echo "DONE"
done

DATASET='ud-english-ewt'
for SEED in 1
do
    echo "nice python3 tree_root_task.py --llm '$MODEL' --dataset '$DATASET' --layer $LAYER --seed $SEED --activation $ACTIVATION"
    nice python3 tree_root_task.py --llm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --activation $ACTIVATION
    wait && echo "DONE"
done

DATASET='ud-english-gum'
for SEED in 1
do
    echo "nice python3 tree_root_task.py --llm '$MODEL' --dataset '$DATASET' --layer $LAYER --seed $SEED --activation $ACTIVATION"
    nice python3 tree_root_task.py --llm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --activation $ACTIVATION
    wait && echo "DONE"
done