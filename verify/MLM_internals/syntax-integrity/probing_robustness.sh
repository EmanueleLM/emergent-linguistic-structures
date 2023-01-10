echo "Starting Bert, Bert-finetuned and RoBERTa experiments..."
sh ./probing_robustness_mlm.sh
wait && echo "DONE"

echo "Starting Glove, Glove-counterfitted and Word2Vec experiments..."
sh ./probing_robustness_w2v.sh
wait && echo "DONE"
