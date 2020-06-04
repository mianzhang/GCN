import argparse

import torch

import gcn

log = gcn.utils.get_logger()


def main(args):
    gcn.utils.set_seed(args.seed)

    log.debug("Loading data from '%s'." % args.data)
    data = gcn.utils.load_pkl(args.data)
    G, vocab = data["graph"], data["vocab"]
    train_samples, dev_samples, test_samples = data["train"], data["dev"], data["test"]
    log.info("Loaded data.")

    A_norm = G.get_adj_norm(args.device)
    X = torch.eye(G.get_node_size())
    log.info("Built A norm.")

    log.debug("Building model...")
    model = gcn.GCN(X.size(-1), vocab["tag"].size(), A_norm, args.hidden_dim).to(args.device)
    opt = gcn.Optim(args.learning_rate, args.max_grad_value, args.weight_decay)
    opt.set_parameters(model.parameters(), args.optimizer)

    model_file = "save/model.pt"

    for name, value in model.named_parameters():
        log.debug("name: {}\t grad: {}".format(name, value.requires_grad))
    nParams = sum([p.nelement() for p in model.parameters()])
    log.debug("number of parameters: %d" % nParams)

    coach = gcn.Coach(model, opt, X, train_samples, dev_samples, test_samples, args)
    if not args.from_begin:
        ckpt = torch.load(model_file)
        coach.load_ckpt(ckpt)
        log.info("Loaded from checkpoint.")

    # Train.
    log.info("Start training...")
    ret = coach.train()

    # Save.
    checkpoint = {
        "best_acc": ret[0],
        "best_epoch": ret[1],
        "best_state": ret[2],
    }
    torch.save(checkpoint, model_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="adjacent.py")
    # data options
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data.")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model.")
    # train options
    parser.add_argument("--from_begin", action="store_true",
                        help="Training from begin.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Computing device.")
    parser.add_argument("--epochs", default=1, type=int,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", default=128, type=int,
                        help="Batch size.")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "rmsprop", "adam"],
                        help="Name of optimizer.")
    parser.add_argument("--learning_rate", type=float, default=0.02,
                        help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="Weight decay.")
    parser.add_argument("--max_grad_value", default=-1, type=float,
                        help="""If the norm of the gradient vector exceeds this,
                        normalize it to have the norm equal to max_grad_norm""")
    parser.add_argument("--drop_rate", type=float, default=0.5,
                        help="Dropout rate.")
    # model options
    parser.add_argument("--hidden_dim", type=int, default=200,
                        help="Dimension of first gcn layer.")
    # others
    parser.add_argument("--seed", type=int, default=24,
                        help="Random seed.")

    args = parser.parse_args()

    main(args)
