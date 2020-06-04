import copy
import time
import random

import torch
from tqdm import tqdm

import gcn

log = gcn.utils.get_logger()


class Coach:

    def __init__(self, model, opt, X, train_samples, dev_samples, test_samples, args):
        self.model = model
        self.opt = opt
        self.X = X
        self.train_samples = train_samples
        self.dev_samples = dev_samples
        self.test_samples = test_samples
        self.device = args.device
        self.args = args

        self.best_acc = None
        self.best_epoch = None
        self.best_state = None

    def load_ckpt(self, ckpt):
        self.best_acc = ckpt["best_acc"]
        self.best_epoch = ckpt["best_epoch"]
        self.best_state = ckpt["best_state"]
        self.model.load_state_dict(self.best_state)

    def train(self):
        log.debug(self.model)

        # Early stopping.
        best_acc, best_epoch, best_state = self.best_acc, self.best_epoch, self.best_state
        # Train
        self.model.train()
        for epoch in tqdm(range(1, self.args.epochs + 1), desc="Training..."):
            self.model.zero_grad()
            self.train_epoch(epoch)

            dev_acc = self.evaluate()
            log.info("[Dev set] [acc: %f]" % dev_acc)
            test_acc = self.evaluate(test=True)
            log.info("[Test set] [acc: %f]" % test_acc)

            if best_acc is None or dev_acc > best_acc:
                best_acc = dev_acc
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                log.info("Save the best model.")

        # The best
        self.model.load_state_dict(best_state)
        log.info("Best in epoch {}:".format(best_epoch))
        dev_acc = self.evaluate()
        log.info("[Dev set] [acc: %f]" % dev_acc)
        test_acc = self.evaluate(test=True)
        log.info("[Test set] [acc: %f]" % test_acc)

        return best_acc, best_epoch, best_state

    def train_epoch(self, epoch):
        # random.shuffle(self.train_samples)
        start_time = time.time()
        epoch_loss = 0

        # to gpu
        X = self.X.to(self.device)
        idxs, tags = gcn.utils.get_idx_tag(self.train_samples)
        idxs = torch.tensor(idxs).long().to(self.device)
        tags = torch.tensor(tags).long().to(self.device)

        nll = self.model.calculate_loss(X, idxs, tags)
        epoch_loss += nll.item()
        nll.backward()
        self.opt.step()

        end_time = time.time()
        log.info("[Epoch %d] [Loss: %f] [Time: %f]" %
                 (epoch, epoch_loss, end_time - start_time))

    def evaluate(self, test=False):
        samples = self.test_samples if test else self.dev_samples
        self.model.eval()
        with torch.no_grad():
            X = self.X.to(self.device)
            idxs, tags = gcn.utils.get_idx_tag(samples)
            idxs = torch.tensor(idxs).long().to(self.device)
            tags = torch.tensor(tags).long()
            pred_tags = self.model(X, idxs).to("cpu")
            acc = torch.mean((pred_tags == tags).float()).item()

        return acc
