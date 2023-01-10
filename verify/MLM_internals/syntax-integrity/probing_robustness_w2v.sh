LAYER='-1'
COPOS=False
ACTIVATION='relu'
for SEED in 42
    do
    ##################################################
    ##################################################

    DATASET='ted'
    MODEL='glove'
    for BUDGET in 1
    do
        echo "$MODEL budget $BUDGET"
        echo "nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False"
        nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False
        wait && echo "DONE"
    done

    MODEL='word2vec'
    for BUDGET in 1
    do
        echo "$MODEL budget $BUDGET"
        echo "nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False"
        nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False
        wait && echo "DONE"
    done

    MODEL='glove-counterfitted'
    for BUDGET in 1
    do
        echo "$MODEL budget $BUDGET"
        echo "nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False"
        nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False
        wait && echo "DONE"
    done

    ##################################################
    ##################################################

    DATASET='en-universal'
    MODEL='glove'
    for BUDGET in 1
    do
        echo "$MODEL budget $BUDGET"
        echo "nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False"
        nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False
        wait && echo "DONE"
    done

    MODEL='word2vec'
    for BUDGET in 1
    do
        echo "$MODEL budget $BUDGET"
        echo "nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False"
        nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False
        wait && echo "DONE"
    done

    MODEL='glove-counterfitted'
    for BUDGET in 1
    do
        echo "$MODEL budget $BUDGET"
        echo "nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False"
        nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False
        wait && echo "DONE"
    done

    ##################################################
    ##################################################

    DATASET='ud-english-pud'
    MODEL='glove'
    for BUDGET in 1
    do
        echo "$MODEL budget $BUDGET"
        echo "nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False"
        nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False
        wait && echo "DONE"
    done

    MODEL='word2vec'
    for BUDGET in 1
    do
        echo "$MODEL budget $BUDGET"
        echo "nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False"
        nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False
        wait && echo "DONE"
    done

    MODEL='glove-counterfitted'
    for BUDGET in 1
    do
        echo "$MODEL budget $BUDGET"
        echo "nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False"
        nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False
        wait && echo "DONE"
    done

    ##################################################
    ##################################################

    DATASET='ud-english-lines'
    MODEL='glove'
    for BUDGET in 1
    do
        echo "$MODEL budget $BUDGET"
        echo "nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False"
        nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False
        wait && echo "DONE"
    done

    MODEL='word2vec'
    for BUDGET in 1
    do
        echo "$MODEL budget $BUDGET"
        echo "nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False"
        nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False
        wait && echo "DONE"
    done

    MODEL='glove-counterfitted'
    for BUDGET in 1
    do
        echo "$MODEL budget $BUDGET"
        echo "nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False"
        nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False
        wait && echo "DONE"
    done

    ##################################################
    ##################################################

    DATASET='ud-english-gum'
    MODEL='glove'
    for BUDGET in 1
    do
        echo "$MODEL budget $BUDGET"
        echo "nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False"
        nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False
        wait && echo "DONE"
    done

    MODEL='word2vec'
    for BUDGET in 1
    do
        echo "$MODEL budget $BUDGET"
        echo "nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False"
        nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False
        wait && echo "DONE"
    done

    MODEL='glove-counterfitted'
    for BUDGET in 1
    do
        echo "$MODEL budget $BUDGET"
        echo "nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False"
        nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False
        wait && echo "DONE"
    done

    ##################################################
    ##################################################

    DATASET='ud-english-ewt'
    MODEL='glove'
    for BUDGET in 1
    do
        echo "$MODEL budget $BUDGET"
        echo "nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False"
        nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False
        wait && echo "DONE"
    done

    MODEL='word2vec'
    for BUDGET in 1
    do
        echo "$MODEL budget $BUDGET"
        echo "nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False"
        nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False
        wait && echo "DONE"
    done

    MODEL='glove-counterfitted'
    for BUDGET in 1
    do
        echo "$MODEL budget $BUDGET"
        echo "nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False"
        nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False
        wait && echo "DONE"
    done
done