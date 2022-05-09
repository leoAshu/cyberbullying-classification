import re
import pandas as pd
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def untokenize_document(document: list) -> str:
    """
    Untokenizes or concats a list of tokens into a string.

    Parameters
    ----------
    document : list
        Input list of string tokens.

    Returns
    -------
    str
        Concatenated tokens.

    """
    return ' '.join(document)


def __get_wordnet_tag_from_treebank_tag(treebank_tag) -> str:
    """
    Converts a treebank tag to corresponding wordnet tag.

    Parameters
    ----------
    treebank_tag : str
        Input treebank tag.

    Returns
    -------
    str
        Corresponding wordnet tag.

    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def preprocess_document(document: str) -> str:
    """
    Performs cleaning, preprocessing, tokenization and lemmatization.

    Uses :func:`~re.sub` to remove special characters.
    Uses :func:`~nltk.corpus.stopwords` to load stopwords.
    Uses :func:`~nltk.tokenize.word_tokenize` to perform tokenization.
    Uses :func:`~nltk.tag.pos_tag` to perform part of speech tagging.
    Uses :func:`~nltk.stem.WordNetLemmatizer` to instantiate a lemmatizer.
    Uses :func:`~nltk.stem.WordNetLemmatizer().lemmatize` to perform lemmatization.

    Parameters
    ----------
    document : str
        Input raw document or text.

    Returns
    -------
    str
        Cleaned and normalized text.

    """
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')

    # convert to lower case
    document = document.lower()

    # remove links, punctuations & other special characters
    document = re.sub(r'http\S+',' ',document)
    document = re.sub('[^a-zA-Z]',' ',document)

    # tokenize text
    document = word_tokenize(document)

    # remove stop words
    document = [word for word in document if word not in stop_words]

    # remove 2 or less letter words
    document = [word for word in document if len(word)>2]

    # perform part of speech tagging on words
    document = pos_tag(document)

    # lemmatize words
    document = [lemmatizer.lemmatize(word,  __get_wordnet_tag_from_treebank_tag(tag)) for word, tag in document]

    # remove 2 or less letter words
    document = [word for word in document if len(word)>2]

    # concatenating the tokens into a string
    document = ' '.join(document)

    return document


def preprocess_corpus(corpus: pd.DataFrame) -> pd.DataFrame:
    """
    Performs cleaning, preprocessing, tokenization and lemmatization on an entire corpus.

    Uses :func:`~preprocess_document` to preprocess a document.
    Uses :func:`~pandas.apply` to :func:`~perform preprocess_document` on the corpus.

    Parameters
    ----------
    corpus : pandas.DataFrame
        Input corpus of raw documents or texts.

    Returns
    -------
    pd.DataFrame
        Output corpus with cleaned documents or texts.

    """
    # preprocessing all texts one at a time
    corpus = corpus.apply(preprocess_document)

    # handle null values if exist
    # replace null value with empty string
    if corpus.isna().sum() > 0:
        corpus = corpus.fillna('')

    return corpus