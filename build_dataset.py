import random
import argparse

import gcn

log = gcn.utils.get_logger()


def main(args):
    gcn.utils.set_seed(args.seed)
    corpus_file = '/'.join(["data", args.dataset, "corpus", args.dataset + ".txt.clean"])
    f = open(corpus_file, 'r')
    corpus = []
    for line in f:
        corpus.append(line.strip())
    f.close()

    train = []
    test = []
    split_file = '/'.join(["data", args.dataset, args.dataset + ".txt"])
    f = open(split_file, 'r')
    for i, line in enumerate(f):
        doc_id, split, label = line.strip().split()
        doc_id = 'O' + str(doc_id)
        if split == "train":
            train.append(doc_id + '\t' + label + '\t' + corpus[i])
        elif split == "test":
            test.append(doc_id + '\t' + label + '\t' + corpus[i])
    f.close()

    random.shuffle(train)
    k = int(len(train) * 0.1)
    train, dev = train[:-k], train[-k:]
    trainset = '/'.join(["data", args.dataset, "dataset", "train.txt"])
    f = open(trainset, 'w')
    f.write('\n'.join(train))
    f.close()
    devset = '/'.join(["data", args.dataset, "dataset", "dev.txt"])
    f = open(devset, 'w')
    f.write('\n'.join(dev))
    f.close()
    testset = '/'.join(["data", args.dataset, "dataset", "test.txt"])
    f = open(testset, 'w')
    f.write('\n'.join(test))
    f.close()

    log.info("number of train docs: {}".format(len(train)))
    log.info("number of dev docs: {}".format(len(dev)))
    log.info("number of test docs: {}".format(len(test)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="build_dataset.py")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=['20ng', 'R8', 'R52', 'ohsumed', 'mr'],
                        help="Dataset name.")
    parser.add_argument("--seed", type=int, default=24,
                        help="Random seed.")
    args = parser.parse_args()

    main(args)
