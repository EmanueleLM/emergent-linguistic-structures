DATASET='ted'
MODEL='bert-finetuned'
LAYER='-9'
for SEED in 1
do
    echo "nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True"
    # nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --specify-accuracy 0.83
    # nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --specify-accuracy 0.86
    nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --specify-accuracy 0.89
    nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --specify-accuracy 0.91
    wait && echo "DONE"
done

DATASET='en-universal'
for SEED in 1
do
    echo "nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True"
    # nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --specify-accuracy 0.83
    # nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --specify-accuracy 0.86
    nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --specify-accuracy 0.89
    nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --specify-accuracy 0.91
    wait && echo "DONE"
done

DATASET='ud-english-pud'
for SEED in 1
do
    echo "nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True"
    # nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --specify-accuracy 0.83
    # nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --specify-accuracy 0.86
    nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --specify-accuracy 0.89
    nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --specify-accuracy 0.91
    wait && echo "DONE"
done

DATASET='ud-english-lines'
for SEED in 1
do
    echo "nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True"
    # nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --specify-accuracy 0.83
    # nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --specify-accuracy 0.86
    nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --specify-accuracy 0.89
    nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --specify-accuracy 0.91
    wait && echo "DONE"
done

DATASET='ud-english-ewt'
for SEED in 1
do
    echo "nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True"
    # nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --specify-accuracy 0.83
    # nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --specify-accuracy 0.86
    nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --specify-accuracy 0.89
    nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --specify-accuracy 0.91
    wait && echo "DONE"
done

DATASET='ud-english-gum'
for SEED in 1
do
    echo "nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True"
    # nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --specify-accuracy 0.83
    # nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --specify-accuracy 0.86
    nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --specify-accuracy 0.89
    nice python3 pos_task.py --mlm $MODEL --dataset $DATASET --layer $LAYER --seed $SEED --classification True --specify-accuracy 0.91
    wait && echo "DONE"
done
