import math
import copy
import argparse

from tqdm import tqdm

import gcn

log = gcn.utils.get_logger()


def read_data(file):
    doc_ids, tags, docs = [], [], []
    f = open(file, 'r')
    for line in f:
        line = line.strip().split()
        doc_id, tag, doc = line[0], line[1], line[2:]
        doc_ids.append(doc_id)
        tags.append(tag)
        docs.append(doc)
    f.close()
    return doc_ids, tags, docs


def make_vocab(doc_ids, tags, docs):
    tag_vocab = gcn.Dict()
    word_vocab = gcn.Dict()
    for lb in tags:
        tag_vocab.add(lb)
    for doc in docs:
        for w in doc:
            word_vocab.add(w)
    node_vocab = copy.deepcopy(word_vocab)
    for doc_id in doc_ids:
        node_vocab.add(doc_id)
    tag_vocab = tag_vocab.prune()
    word_vocab = word_vocab.prune()
    node_vocab = node_vocab.prune()
    log.info(tag_vocab.label2idx)
    log.info("num of words: {}".format(word_vocab.size()))

    return {"tag": tag_vocab, "word": word_vocab, "node": node_vocab}


def make_samples(doc_ids, tag_ids):
    ret = []
    for x, y in zip(doc_ids, tag_ids):
        ret.append(gcn.Sample(x, y))

    return ret


def main(args):
    train_doc_ids, train_tags, train_docs = read_data(args.train)
    dev_doc_ids, dev_tags, dev_docs = read_data(args.dev)
    test_doc_ids, test_tags, test_docs = read_data(args.test)
    all_doc_ids = train_doc_ids + dev_doc_ids + test_doc_ids
    all_docs = train_docs + dev_docs + test_docs

    vocab = make_vocab(all_doc_ids, train_tags, all_docs)
    words = list(vocab["word"].label2idx.keys())
    G = gcn.Graph(vocab["node"].size())

    # PMI
    fre = {}
    fre_pair = {}
    windows = []
    for doc in all_docs:
        if len(doc) <= args.window_size:
            windows.append(doc)
            continue
        for i in range(len(doc) - args.window_size + 1):
            windows.append(doc[i: i + args.window_size])
    for window in tqdm(windows, desc="fre and fre_pair"):
        window = list(set(window))
        for i in range(len(window)):
            x = window[i]
            if x not in fre:
                fre[x] = 1
            else:
                fre[x] += 1
            for j in range(i + 1, len(window)):
                y = window[j]
                pair = x + '-' + y if x < y else y + '-' + x
                if pair not in fre_pair:
                    fre_pair[pair] = 1
                else:
                    fre_pair[pair] += 1

    PMI = {}
    for i in tqdm(range(len(words)), desc="PMI"):
        x = words[i]
        pmi_x = fre[x] / len(windows)
        for j in range(i + 1, len(words)):
            y = words[j]
            pmi_y = fre[y] / len(windows)
            pair = x + '-' + y if x < y else y + '-' + x
            pmi_pair = fre_pair[pair] / len(windows) if pair in fre_pair else 0
            PMI[pair] = math.log(pmi_pair / (pmi_x * pmi_y)) if pmi_pair > 0 else 0

    # TF-IDF
    term_fre = {}
    for doc_id, doc in tqdm(zip(all_doc_ids, all_docs),
                            desc="term_fre", total=len(all_docs)):
        for w in doc:
            td = w + '-' + doc_id
            if td not in term_fre:
                term_fre[td] = 1
            else:
                term_fre[td] += 1
    TF = {}
    for w in tqdm(words, desc="TF"):
        for doc_id in all_doc_ids:
            td = w + '-' + doc_id
            tf = math.log(term_fre[td] + 1 if td in term_fre else 1)
            TF[td] = tf

    doc_fre = {}
    for doc in tqdm(all_docs, desc="doc_fre"):
        doc = list(set(doc))
        for w in doc:
            if w not in doc_fre:
                doc_fre[w] = 1
            else:
                doc_fre[w] += 1
    IDF = {}
    for w in tqdm(words, desc="IDF"):
        idf = math.log(len(all_docs) / doc_fre[w])
        IDF[w] = idf

    TF_IDF = {}
    for w in tqdm(words, desc="TF_IDF"):
        idf = IDF[w]
        for doc_id in all_doc_ids:
            td = w + '-' + doc_id
            tf = TF[td]
            TF_IDF[td] = tf * idf

    # add edges
    for w in tqdm(words, desc="add word-doc edge"):
        for doc_id in all_doc_ids:
            td = w + '-' + doc_id
            if TF_IDF[td] == 0:
                continue
            G.add_edge(vocab["node"].lookup(w), vocab["node"].lookup(doc_id), TF_IDF[td])

    for i in tqdm(range(len(words)), desc="add word-word edge"):
        x = words[i]
        for j in range(i + 1, len(words)):
            y = words[j]
            pair = x + '-' + y if x < y else y + '-' + x
            if PMI[pair] <= 0:
                continue
            else:
                G.add_edge(vocab["node"].lookup(x), vocab["node"].lookup(y), PMI[pair])

    log.info("num of nodes: {}".format(G.get_node_size()))
    log.info("num of edges: {}".format(G.get_edge_size()))

    train_node_ids = [vocab["node"].lookup(x) for x in train_doc_ids]
    dev_node_ids = [vocab["node"].lookup(x) for x in dev_doc_ids]
    test_node_ids = [vocab["node"].lookup(x) for x in test_doc_ids]
    train_tag_ids = [vocab["tag"].lookup(x) for x in train_tags]
    dev_tag_ids = [vocab["tag"].lookup(x) for x in dev_tags]
    test_tag_ids = [vocab["tag"].lookup(x) for x in test_tags]

    train_samples = make_samples(train_node_ids, train_tag_ids)
    dev_samples = make_samples(dev_node_ids, dev_tag_ids)
    test_samples = make_samples(test_node_ids, test_tag_ids)

    data = {"graph": G, "vocab": vocab,
            "train": train_samples,
            "dev": dev_samples,
            "test": test_samples}
    gcn.utils.save_pkl(data, args.data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="adjacent.py")
    parser.add_argument("--train", type=str, required=True,
                        help="Path to train set.")
    parser.add_argument("--dev", type=str, required=True,
                        help="Path to dev set.")
    parser.add_argument("--test", type=str, required=True,
                        help="Path to test set.")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to save data.")
    parser.add_argument("--window_size", type=int, default=20,
                        help="Size of sliding window.")
    args = parser.parse_args()

    main(args)
