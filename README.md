## Emergent Linguistic Structure in Neural Networks are Fragile

We provide a *very* preliminary version of the code to replicate all the experiments of the paper
`Emergent Linguistic Structure in Neural Networks are Fragile'. A draft of the article is available here:
https://arxiv.org/pdf/2210.17406.pdf

If you want to run the experiments to prove the robustness of a neural network on the 4 probing tasks, cd to the 
verify/MLM_internals/syntax-integrity (you can ignore the folders structure for now, it will be enhanced soon) folder and run
```
python3 probing_robustness.py
```

This will launch a program that:
- collects the representations for a dataset and an embedding/LLM; 
- train a probe on each of the 4 probing tasks; 
- measure the robustness of the probe against a perturbation model;
- print and store the results.

## !!warning!! All the representations will be pre-computed and stored inside the data/datasets/conll folder as numpy files, so please consider that 
## you'll need a lot of space if you decide to run all the experiments of the paper (approximately 150 GB).

Please look at the argparse of probing_robustness.py for all the options available, there are *a lot* of choices, among those:
- select between Word2Vec, GloVe, BERT and RoBERTa (and in the case of BERT and RoBERTa, the hidden layer to use to extract the representation of the input and the size of the LLM)
- select among 6 datasets
- select the kind of probing network that learns the syntactic task
- select the perturbation type (coPOS, coCO)
- select the perturbation budget (the number of words perturbed per sentence)
- select the robustness scenario (between average worst case and average case)
- ...

Soon, we will publish an enhanced version of both the code and the readme that allows for straightforward execution of the experiments.
If you need a bash file to automate the experiments or to look at all the various combinations of experiments you can run, please refer to
probing_task_mlm.sh or probing_task_w2v.sh

If you want to cite the article, please use the following bibtex:
```
@article{la2022emergent,
  title={Emergent Linguistic Structures in Neural Networks are Fragile},
  author={La Malfa, Emanuele and Wicker, Matthew and Kiatkowska, Marta},
  journal={arXiv preprint arXiv:2210.17406},
  year={2022}
}
```

If you need any help with the code, especially regarding the instructions to execute a specific instance of an experiment, please consider to open an issue or write directly to
my email address (see my github main page).
