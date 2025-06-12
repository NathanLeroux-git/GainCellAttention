# Evaluation

Scripts that implement the evaluation routine.

Additional requirements:

```bash
pip install lm-eval
pip install spacy
```

In order to run the GPT-2 style evaluation on LAMBADA, execute `` with the corresponding model name, e.g.:

```bash
python3 eval_lambada_eval.py --model_name openai-community/gpt2-xl
```

The model name can either be a huggingface model name, or one the custom finetuned gain-cell models. For those you have to make sure that the weights are present in the required directory.

To execute the evaluation suite for all models on the other datasets provided in the paper, run:

```bash
bash run_lm_eval.sh
```
