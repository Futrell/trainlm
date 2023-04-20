import sys
import itertools

from datasets import load_dataset
from transformers import AutoTokenizer

import preprocess

MODEL = "gpt2"
VOCAB_SIZE = 52000


def train_tokenizer(dataset, subset):
    data = load_dataset(dataset, subset, beam_runner="DirectRunner")    
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    text = itertools.chain.from_iterable(map(preprocess.preprocess_text, data['train']['text']))
    new_tokenizer = tokenizer.train_new_from_iterator(text, VOCAB_SIZE)
    tokenizer_path = "_".join(["tokenizer", dataset, subset])
    new_tokenizer.save_pretrained(tokenizer_path)
    return new_tokenizer

if __name__ == '__main__':
    train_tokenizer(*sys.argv[1:])

    


    

    
