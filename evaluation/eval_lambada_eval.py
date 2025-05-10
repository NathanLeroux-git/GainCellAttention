#!/usr/bin/env python
# Evaluate GPT-2 model on lambada dataset.
# Inspired by discussion at:
# https://github.com/openai/gpt-2/issues/131#issuecomment-497136199 
# https://github.com/huggingface/transformers/issues/491 
# https://github.com/EleutherAI/lm-evaluation-harness/issues/350
# Implementation based on:
# https://github.com/cybertronai/bflm/blob/d58a6860451ee2afa3688aff13d104ad74001ebe/eval_lambada_slow.py#L77

import os
import math
import argparse
import torch
from transformers import AutoModel, AutoTokenizer, GPT2LMHeadModel
from datasets import load_dataset
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--print-every-n',  type=int, default=100, help='print results every n lines')
parser.add_argument('--beam-width',  type=int, default=1024, help='predict this many results before stopword filtering')
parser.add_argument('--model-name',  type=str, default="gpt2", help='predict this many results before stopword filtering')
parser.add_argument('--output-path',  type=str, default="./results/lambada_results_spacy.txt", help='predict this many results before stopword filtering')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using ", device)
args.device = device

model_name = args.model_name

if not "dram" in model_name:
    enc = AutoTokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(device) 
else:
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

    from modules.model_gpt_eval import DRAM_GPT2_MODEL_DICT

    additional_suffix = { # hardcoded due to our model names
        "dram-gpt2-xl-DRAMAttention": "_1000_iters",
        "dram-gpt2-xl-LinearDRAMAttention": "_10000_iters"
    }

    enc = AutoTokenizer.from_pretrained("gpt2")
    model_custom = DRAM_GPT2_MODEL_DICT[model_name].from_pretrained(model_name, dtype=torch.float32, batch_size_model=1, additional_suffix=additional_suffix.get(model_name, None))
    model_custom.to(device)

    def forward_custom(line_encoded, past_key_values=None, use_cache=True):
        if past_key_values is None:
            input_idxs = line_encoded
        else:
            input_idxs = torch.concat([line_encoded, past_key_values], dim=-1)

        retval = model_custom(input_idxs)
        return {"logits": retval.logits, "past_key_values": input_idxs}

    model = forward_custom

dataset_name = "EleutherAI/lambada_openai"
dataset = load_dataset(dataset_name, split="test") 

from spacy.lang.en.stop_words import STOP_WORDS as stopwords

# stopwords = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'}


def argmax(t):
    return int(torch.argmax(t).item())


def to_list(tensor):
    return list(tensor.cpu().numpy())


def remove_last_word(line):
    line = line.strip()
    words = line.replace("\n", " ").split(' ')
    length_of_words = sum([len(w) for w in words[:-1]])
    cutpos = length_of_words+len(words)-2
    text = line[:cutpos]
    target = line[cutpos:]
    assert len(target) > 0
    return text, target


def remove_last_tok(line):
    toks = enc.tokenize(line)
    length_of_word = len(toks[-1])
    assert length_of_word > 0
    return toks[:-1], toks[-1]



def predict(full_line, line, max_predictions, target):
    """Give continuation of the line with at most max_predictions BPE tokens. Returns line extended with predictions of
     the model."""
    line_encoded = enc.encode(line, return_tensors='pt')
    line_encoded = line_encoded[0].unsqueeze_(0) # batch of size 1
    line_encoded_list = list(line_encoded[0].numpy())
    line_encoded_list_pred = []
    line_encoded = line_encoded.to(device)
    state = None
    total = 0

    for i in range(max_predictions):
        output = model(line_encoded, past_key_values=state, use_cache=True)
        logits = output["logits"]
        state = output["past_key_values"]
        _, line_encoded_candidates = torch.topk(logits[:,-1,:], k=args.beam_width, dim=-1)

        # determine which candidates are stopwords by decoding them and
        # comparing against NLTK stopword list
        line_encoded_candidates = to_list(line_encoded_candidates[0])
        is_stopword = []
        for tok in line_encoded_candidates:
            decoded = enc.decode([tok.item()]).strip()

            is_stop = decoded in stopwords
            is_stopword.append(is_stop)

        # find first prediction which is not a stopword
        predicted = None
        for (idx, candidate) in enumerate(line_encoded_candidates):
            if is_stopword[idx]:
                # print('skipping stopword ', idx)
                continue
            else:
                predicted = candidate
                break
        assert predicted is not None
        line_encoded = torch.tensor([[predicted]]).to(device)
        line_encoded_list_pred.append(predicted)
        
        # if i < len(target):
        if i < 1: # Use only first token of last word
            # # set is_stop logits to min float value
            # logits[0, -1, line_encoded_candidates] -= torch.tensor(is_stopword) -1e10
            
            log_probs = torch.nn.functional.log_softmax(logits[0, -1, :], dim=-1)
            probs = torch.nn.functional.softmax(logits[0, -1, :], dim=-1)

            log_ppl = log_probs[target[i]].item()
            prob_target = probs[target[i]].item()
            
            total += 1

        decoded_full = enc.decode(line_encoded_list_pred)
        
        if len(decoded_full.strip().split(" ")) > 1:
            break
        if "." in decoded_full:
            break

    line_encoded_list += line_encoded_list_pred
    return enc.decode(line_encoded_list), enc.decode(line_encoded_list_pred), line_encoded_list_pred, log_ppl/total


import re

def main():
    errors = 0
    total = 0
    total_toks = 0
    num_zeros = 0
    log_ppl = 0.0
    for i, example in enumerate(tqdm(dataset, desc="Processing", unit="lines")):
        line = example["text"]
        line = line.strip()
        context, last_word = remove_last_word(line)
        
        last_tok = enc.encode(last_word)
        last_word = enc.decode(last_tok[:1]).replace("\n", " ").strip()

        if last_word in stopwords or len(re.sub('\W+','', last_word))==0:
            continue

        # because BPE tokens can span words, predict several BPE tokens
        # and then identify the single word
        prediction, prediction_only, pred_tokens, log_ppl_i  = predict(example["text"], context, 3, last_tok)
        
        log_ppl -= log_ppl_i
        # string generated by the model
        predicted_part = prediction[len(context):].strip()

        # first word in the generated string
        predicted_word = prediction_only.replace("\n", " ").strip().split(' ')[0]
       
        # remove special characters for comparison
        predicted_word = re.sub('\W+','', predicted_word)

        len_of_comp = min(len(predicted_word), len(last_word))
        pred_to_use = predicted_word[:len_of_comp]
        last_to_use = last_word[:len_of_comp]

        is_error = pred_to_use != last_to_use
        # if prediction reduced to empty strin (e.g. only contained specail characters) count it as error
        if len_of_comp == 0:
            is_error = True
            num_zeros += 1 

        if is_error:
            errors += 1
        total+=1
        total_toks += len(last_tok)

        predictions_file.write(f"{line}\n{predicted_word}\n{is_error}\n\n")

        if i%args.print_every_n == 0:
            print(f"{i:5d} acc: {1-errors/total:.4f}, log ppl: {log_ppl/total:.4f}, ppl: {math.exp(log_ppl/total):.4f}")

    print("Final accuracy")
    print(f"acc: {1-errors/total:.4f}")
    print(f"log_ppl: {log_ppl/total:.4f}")
    print(f"ppl: {math.exp(log_ppl/total):.4f}")
    print("num_zeros", num_zeros)
    print("total", total)
    print("total_toks", total_toks)

    
    if not os.path.exists(args.output_path):
        line = "# model_name, acc, ppl, log_ppl, num_zeros, total, total_toks\n"
        with open(args.output_path, "w") as f:
            f.write(line)

    line = f"{model_name}, {1-errors/total:.6f}, {math.exp(log_ppl/total):.6f}, {log_ppl/total:.6f}, {num_zeros}, {total}, {total_toks}\n"
    with open(args.output_path, "a") as f:
        f.write(line)


if __name__=='__main__':
    main()
