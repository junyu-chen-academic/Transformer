import os
import spacy
import torch
from torchtext.datasets import Multi30k
from  torchtext.vocab import build_vocab_from_iterator

###########################################################
###########################################################
def yield_tokens(data_iter, tokenizer, index: int):
    """
    Return the tokens for the appropriate language.

    Args:
        data_iter:    text here 
        tokenizer:    tokenizer for the language
        index:        index of the language in the tuple | (de=0, en=1)
        
    Yields:
        sequences based on index       
    """
    for from_tuple in data_iter:
        yield tokenizer(from_tuple[index])

###########################################################
###########################################################
def build_vocabulary(spacy_de, spacy_en, min_freq: int = 2):
  
    def tokenize_de(text: str):
        """
            Call the German tokenizer.

            Args:
                text:         string 
                min_freq:     minimum frequency needed to include a word in the vocabulary
            
            Returns:
                tokenized list of strings       
        """
        return tokenize(text, spacy_de)

    def tokenize_en(text: str):
        """
            Call the English tokenizer.

            Args:
                text:         string 
            
            Returns:
                tokenized list of strings       
        """
        return tokenize(text, spacy_en)

    print("Building German Vocabulary...")

    # load train, val, and test data pipelines
    train, val, test = Multi30k(language_pair=("de", "en"))

    # generate source vocabulary
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_de, index=0), # tokens for each German sentence (index 0)
        min_freq=min_freq, 
        specials=["<bos>", "<eos>", "<pad>", "<unk>"],
    )

    print("Building English Vocabulary...")

    # generate target vocabulary
    vocab_trg = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_en, index=1), # tokens for each English sentence (index 1)
        min_freq=2, # 
        specials=["<bos>", "<eos>", "<pad>", "<unk>"],
    )

    # set default token for out-of-vocabulary words (OOV)
    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_trg.set_default_index(vocab_trg["<unk>"])

    return vocab_src, vocab_trg

###########################################################
###########################################################
def tokenize(text: str, tokenizer):
    """
    Split a string into its tokens using the provided tokenizer.

    Args:
        text:         string 
        tokenizer:    tokenizer for the language
        
    Returns:
        tokenized list of strings       
    """
    return [tok.text.lower() for tok in tokenizer.tokenizer(text)]

###########################################################
###########################################################
def load_tokenizers():
    """
    Load the German and English tokenizers provided by spaCy.

    Returns:
        spacy_de:     German tokenizer
        spacy_en:     English tokenizer
    """
    try:
        spacy_de = spacy.load("de_core_news_sm")
    except OSError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except OSError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    print("Loaded English and German tokenizers.")
    return spacy_de, spacy_en

###########################################################
###########################################################
def load_vocab(spacy_de, spacy_en, min_freq: int = 2):
    """
    Args:
        spacy_de:     German tokenizer
        spacy_en:     English tokenizer
        min_freq:     minimum frequency needed to include a word in the vocabulary

    Returns:
        vocab_src:    German vocabulary
        vocab_trg:     English vocabulary       
    """

    if not os.path.exists("vocab.pt"):
        # build the German/English vocabulary if it does not exist
        vocab_src, vocab_trg = build_vocabulary(spacy_de, spacy_en, min_freq)
        # save it to a file
        torch.save((vocab_src, vocab_trg), "vocab.pt")
    else:
        # load the vocab if it exists
        vocab_src, vocab_trg = torch.load("vocab.pt")

    print("Finished.\nVocabulary sizes:")
    print("\tSource:", len(vocab_src))
    print("\tTarget:", len(vocab_trg))
    return vocab_src, vocab_trg

# global variables used later in the script
spacy_de, spacy_en = load_tokenizers()
vocab_src, vocab_trg = load_vocab(spacy_de, spacy_en)