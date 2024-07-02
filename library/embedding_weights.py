import argparse
import torch
import torch.nn.functional as F

def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--pinned_terms",
        type=str,
        default=[],
        nargs="*",
    )
    parser.add_argument(
        "--positive_captions",
        type=str,
        default=[],
        nargs="*",
    )
    parser.add_argument(
        "--negative_captions",
        type=str,
        default=[],
        nargs="*",
    )


class EmbeddingLossManager:
    def __init__(self, embed, dataloader, pinned_terms):
        self.embed = embed
        self.dataloader = dataloader
        self.pinned_terms = pinned_terms
        with torch.no_grad():
            self.cache = self.__cache_embeds()
            self.pinned_terms_orig = embed(pinned_terms).detach()

    def make_embeds(self, captions):
        return torch.cat([self.embed([c]) for c in captions]).sum(dim=0, keepdims=True)

    def __cache_embeds(self):
        memo = {}
        for batch in self.dataloader:
            for entry_terms in batch.get("positive_captions", []) + batch.get("negative_captions", []):
                key = "|".join(entry_terms)
                if key in memo:
                    continue
                memo[key] = self.make_embeds(entry_terms)
        return memo

    def get_term_similarity(self, batch, key):
        keys = ["|".join(k) for k in batch[key]]
        b_size = len(batch["captions"])
        q = torch.stack([self.cache[k] for k in keys]).view(b_size, -1)
        p = self.embed(batch["captions"]).reshape(b_size, -1)
        return F.cosine_similarity(q, p, dim=1)

    def get_positive_loss(self, batch):
        return 1 - self.get_term_similarity(batch, "positive_captions")

    def get_negative_loss(self, batch):
        return 1 + self.get_term_similarity(batch, "negative_captions")

    def get_pinned_loss(self):
        q = self.pinned_terms_orig.view(len(self.pinned_terms), -1)
        p = self.embed(self.pinned_terms).view(len(self.pinned_terms), -1)
        return (1 - F.cosine_similarity(q, p, dim=1))
