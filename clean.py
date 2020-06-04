import re
import argparse
import ssl

from nltk.corpus import stopwords
import nltk

import gcn

log = gcn.utils.get_logger()


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def main(args):
    # load stop words
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    log.info(stop_words)

    corpus_file = '/'.join(["data", args.dataset, "corpus", args.dataset + ".txt"])
    doc_content_list = []
    f = open(corpus_file, 'r')
    for line in f.readlines():
        doc_content_list.append(line.strip())
    f.close()

    word_freq = {}  # to remove rare words

    for doc_content in doc_content_list:
        temp = clean_str(doc_content)
        words = temp.split()
        for word in words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

    clean_docs = []
    for doc_content in doc_content_list:
        temp = clean_str(doc_content)
        words = temp.split()
        doc_words = []
        for word in words:
            # word not in stop_words and word_freq[word] >= 5
            if args.dataset == 'mr':
                doc_words.append(word)
            elif word not in stop_words and word_freq[word] >= 5:
                doc_words.append(word)

        doc_str = ' '.join(doc_words).strip()
        clean_docs.append(doc_str)

    clean_corpus_str = '\n'.join(clean_docs)

    f = open(corpus_file + ".clean", 'w')
    f.write(clean_corpus_str)
    f.close()

    min_len = 10000
    aver_len = 0
    max_len = 0

    f = open(corpus_file + ".clean", 'r')
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        temp = line.split()
        aver_len = aver_len + len(temp)
        if len(temp) < min_len:
            min_len = len(temp)
        if len(temp) > max_len:
            max_len = len(temp)
    f.close()
    aver_len = 1.0 * aver_len / len(lines)
    log.info('min_len : ' + str(min_len))
    log.info('max_len : ' + str(max_len))
    log.info('average_len : ' + str(aver_len))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="clean.py")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=['20ng', 'R8', 'R52', 'ohsumed', 'mr'],
                        help="Dataset name.")
    args = parser.parse_args()

    main(args)
