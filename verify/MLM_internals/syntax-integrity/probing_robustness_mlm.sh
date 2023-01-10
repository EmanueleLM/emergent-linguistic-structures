COPOS=False
ACTIVATION='relu'
for LAYER in '-9' '-5'
do
    for SEED in 42
        do
        DATASET='ted'
        MODEL='bert'
        for BUDGET in 1
        do
            echo "$MODEL budget $BUDGET"
            echo "nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False"
            nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False
            wait && echo "DONE"
        done

        # MODEL='bert-finetuned'
        # for BUDGET in 1
        # do
        #     echo "$MODEL budget $BUDGET"
        #     echo "nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False"
        #     # nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False --specify-accuracy 0.89
        #     # nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False --specify-accuracy 0.91
        #     wait && echo "DONE"
        # done

        MODEL='roberta'
        for BUDGET in 1
        do
            echo "$MODEL budget $BUDGET"
            echo "nice python3 probing_robustness.py --llm '$MODEL' --dataset '$DATASET' --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario 'worst' --lp-norm 2 --wordnet-mode False"
            nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False
            wait && echo "END"
        done

        ##################################################
        ##################################################

        DATASET='en-universal'
        MODEL='bert'
        for BUDGET in 1
        do
            echo "$MODEL budget $BUDGET"
            echo "nice python3 probing_robustness.py --llm '$MODEL' --dataset '$DATASET' --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario 'worst' --lp-norm 2 --wordnet-mode False"
            nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False
            wait && echo "DONE"
        done

        # MODEL='bert-finetuned'
        # for BUDGET in 1
        # do
        #     echo "$MODEL budget $BUDGET"
        #     echo "nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False"
        #     # nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False --specify-accuracy 0.89
        #     # nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False --specify-accuracy 0.91
        #     wait && echo "DONE"
        # done

        MODEL='roberta'
        for BUDGET in 1
        do
            echo "$MODEL budget $BUDGET"
            echo "nice python3 probing_robustness.py --llm '$MODEL' --dataset '$DATASET' --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario 'worst' --lp-norm 2 --wordnet-mode False"
            nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False
            wait && echo "DONE"
        done

        ##################################################
        ##################################################

        DATASET='ud-english-pud'
        MODEL='bert'
        for BUDGET in 1
        do
            echo "$MODEL budget $BUDGET"
            echo "nice python3 probing_robustness.py --llm '$MODEL' --dataset '$DATASET' --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario 'worst' --lp-norm 2 --wordnet-mode False"
            nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False
            wait && echo "DONE"
        done

        # MODEL='bert-finetuned'
        # for BUDGET in 1
        # do
        #     echo "$MODEL budget $BUDGET"
        #     echo "nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False"
            # nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False --specify-accuracy 0.89
        #     # nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False --specify-accuracy 0.91
        #     wait && echo "DONE"
        # done

        MODEL='roberta'
        for BUDGET in 1
        do
            echo "$MODEL budget $BUDGET"
            echo "nice python3 probing_robustness.py --llm '$MODEL' --dataset '$DATASET' --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario 'worst' --lp-norm 2 --wordnet-mode False"
            nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False
            wait && echo "DONE"
        done

        ##################################################
        ##################################################

        DATASET='ud-english-lines'
        MODEL='bert'
        for BUDGET in 1
        do
            echo "$MODEL budget $BUDGET"
            echo "nice python3 probing_robustness.py --llm '$MODEL' --dataset '$DATASET' --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario 'worst' --lp-norm 2 --wordnet-mode False"
            nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False
            wait && echo "DONE"
        done

        # MODEL='bert-finetuned'
        # for BUDGET in 1
        # do
        #     echo "$MODEL budget $BUDGET"
        #     echo "nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False"
        #     nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False --specify-accuracy 0.89
        #     nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False --specify-accuracy 0.91
        #     wait && echo "DONE"
        # done

        MODEL='roberta'
        for BUDGET in 1
        do
            echo "$MODEL budget $BUDGET"
            echo "nice python3 probing_robustness.py --llm '$MODEL' --dataset '$DATASET' --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario 'worst' --lp-norm 2 --wordnet-mode False"
            nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False
            wait && echo "DONE"
        done

        ##################################################
        ##################################################

        DATASET='ud-english-gum'
        MODEL='bert'
        for BUDGET in 1
        do
            echo "$MODEL budget $BUDGET"
            echo "nice python3 probing_robustness.py --llm '$MODEL' --dataset '$DATASET' --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario 'worst' --lp-norm 2 --wordnet-mode False"
            nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False
            nice python3 probing_robustness.py --llm '$MODEL' --dataset '$DATASET2' --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario 'worst' --lp-norm 2 --wordnet-mode False
            wait && echo "DONE"
        done

        # MODEL='bert-finetuned'
        # for BUDGET in 1
        # do
        #     echo "$MODEL budget $BUDGET"
        #     echo "nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False
        #     nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False --specify-accuracy 0.89
        #     nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False --specify-accuracy 0.91
        #     wait && echo "DONE"
        # done

        MODEL='roberta'
        for BUDGET in 1
        do
            echo "$MODEL budget $BUDGET"
            echo "nice python3 probing_robustness.py --llm '$MODEL' --dataset '$DATASET' --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario 'worst' --lp-norm 2 --wordnet-mode False"
            nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False
            wait && echo "DONE"
        done

        ##################################################
        ##################################################

        DATASET='ud-english-ewt'
        MODEL='bert'
        for BUDGET in 1
        do
            echo "$MODEL budget $BUDGET"
            echo "nice python3 probing_robustness.py --llm '$MODEL' --dataset '$DATASET' --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario 'worst' --lp-norm 2 --wordnet-mode False"
            nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False
            nice python3 probing_robustness.py --llm '$MODEL' --dataset '$DATASET2' --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario 'worst' --lp-norm 2 --wordnet-mode False
            wait && echo "DONE"
        done

        # MODEL='bert-finetuned'
        # for BUDGET in 1
        # do
        #     echo "$MODEL budget $BUDGET"
        #     echo "nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False"
        #     nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False --specify-accuracy 0.89
        #     nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False --specify-accuracy 0.91
        #     wait && echo "DONE"
        # done

        MODEL='roberta'
        for BUDGET in 1
        do
            echo "$MODEL budget $BUDGET"
            echo "nice python3 probing_robustness.py --llm '$MODEL' --dataset '$DATASET' --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario 'worst' --lp-norm 2 --wordnet-mode False"
            nice python3 probing_robustness.py --seed $SEED --llm $MODEL --layer-llm $LAYER --dataset $DATASET --activationp1 $ACTIVATION --activationt1 $ACTIVATION --activationr1 $ACTIVATION --activationd1 $ACTIVATION --perturbation-budget $BUDGET --copos $COPOS --perturbation-scenario worst --lp-norm 2 --wordnet-mode False
            wait && echo "DONE"
        done
    done
done