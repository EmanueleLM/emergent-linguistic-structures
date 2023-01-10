MODEL='bert'
ACTIVATION='relu'
LAYER='-9'
DATASET='ted'
for SEED in 1
do
    echo "nice python3 structural_probe_task.py --llm '$MODEL' --dataset '$DATASET' --layer $LAYER --seed $SEED --activation $ACTIVATION --classification true"
    nice python3 structural_probe_task.py --llm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --activation $ACTIVATION --classification true
    wait && echo "DONE"
done

DATASET='en-universal'
for SEED in 1
do
    echo "nice python3 structural_probe_task.py --llm '$MODEL' --dataset '$DATASET' --layer $LAYER --seed $SEED --activation $ACTIVATION --classification true"
    nice python3 structural_probe_task.py --llm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --activation $ACTIVATION --classification true
    wait && echo "DONE"
done

DATASET='ud-english-pud'
for SEED in 1
do
    echo "nice python3 structural_probe_task.py --llm '$MODEL' --dataset '$DATASET' --layer $LAYER --seed $SEED --activation $ACTIVATION --classification true"
    nice python3 structural_probe_task.py --llm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --activation $ACTIVATION --classification true
    wait && echo "DONE"
done

DATASET='ud-english-lines'
for SEED in 1
do
    echo "nice python3 structural_probe_task.py --llm '$MODEL' --dataset '$DATASET' --layer $LAYER --seed $SEED --activation $ACTIVATION --classification true"
    nice python3 structural_probe_task.py --llm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --activation $ACTIVATION --classification true
    wait && echo "DONE"
done

DATASET='ud-english-ewt'
for SEED in 1
do
    echo "nice python3 structural_probe_task.py --llm '$MODEL' --dataset '$DATASET' --layer $LAYER --seed $SEED --activation $ACTIVATION --classification true"
    nice python3 structural_probe_task.py --llm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --activation $ACTIVATION --classification true
    wait && echo "DONE"
done

DATASET='ud-english-gum'
for SEED in 1
do
    echo "nice python3 structural_probe_task.py --llm '$MODEL' --dataset '$DATASET' --layer $LAYER --seed $SEED --activation $ACTIVATION --classification true"
    nice python3 structural_probe_task.py --llm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --activation $ACTIVATION --classification true
    wait && echo "DONE"
done

###############################################################
###############################################################
###############################################################

DATASET='ted'
MODEL='roberta'
for SEED in 1
do
    echo "nice python3 structural_probe_task.py --llm '$MODEL' --dataset '$DATASET' --layer $LAYER --seed $SEED --activation $ACTIVATION --classification true"
    nice python3 structural_probe_task.py --llm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --activation $ACTIVATION --classification true
    wait && echo "DONE"
done

DATASET='en-universal'
for SEED in 5
do
    echo "nice python3 structural_probe_task.py --llm '$MODEL' --dataset '$DATASET' --layer $LAYER --seed $SEED --activation $ACTIVATION --classification true"
    nice python3 structural_probe_task.py --llm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --activation $ACTIVATION --classification true
    wait && echo "DONE"
done

DATASET='ud-english-pud'
for SEED in 1
do
    echo "nice python3 structural_probe_task.py --llm '$MODEL' --dataset '$DATASET' --layer $LAYER --seed $SEED --activation $ACTIVATION --classification true"
    nice python3 structural_probe_task.py --llm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --activation $ACTIVATION --classification true
    wait && echo "DONE"
done

DATASET='ud-english-lines'
for SEED in 1
do
    echo "nice python3 structural_probe_task.py --llm '$MODEL' --dataset '$DATASET' --layer $LAYER --seed $SEED --activation $ACTIVATION --classification true"
    nice python3 structural_probe_task.py --llm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --activation $ACTIVATION --classification true
    wait && echo "DONE"
done

DATASET='ud-english-ewt'
for SEED in 1
do
    echo "nice python3 structural_probe_task.py --llm '$MODEL' --dataset '$DATASET' --layer $LAYER --seed $SEED --activation $ACTIVATION --classification true"
    nice python3 structural_probe_task.py --llm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --activation $ACTIVATION --classification true
    wait && echo "DONE"
done

DATASET='ud-english-gum'
for SEED in 1
do
    echo "nice python3 structural_probe_task.py --llm '$MODEL' --dataset '$DATASET' --layer $LAYER --seed $SEED --activation $ACTIVATION --classification true"
    nice python3 structural_probe_task.py --llm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --activation $ACTIVATION --classification true
    wait && echo "DONE"
done