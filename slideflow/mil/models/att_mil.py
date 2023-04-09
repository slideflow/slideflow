import torch
from torch import nn
from typing import Optional

# -----------------------------------------------------------------------------

class Attention_MIL(nn.Module):
    """Attention-based multiple instance learning model.

    Implementation from: https://github.com/KatherLab/marugoto

    """
    def __init__(
        self,
        n_feats: int,
        n_out: int,
        encoder: Optional[nn.Module] = None,
        attention: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
    ) -> None:
        """Create a new attention MIL model.
        Args:
            n_feats:  The number of features each bag instance has.
            n_out:  The number of output layers of the model.
            encoder:  A network transforming bag instances into feature vectors.
        """
        super().__init__()
        self.encoder = encoder or nn.Sequential(nn.Linear(n_feats, 256), nn.ReLU())
        self.attention = attention or Attention(256)
        self.head = head or nn.Sequential(
            nn.Flatten(), nn.BatchNorm1d(256), nn.Dropout(), nn.Linear(256, n_out)
        )

    def forward(self, bags, lens):
        assert bags.ndim == 3
        assert bags.shape[0] == lens.shape[0]

        embeddings = self.encoder(bags)

        masked_attention_scores = self._masked_attention_scores(embeddings, lens)
        weighted_embedding_sums = (masked_attention_scores * embeddings).sum(-2)

        scores = self.head(weighted_embedding_sums)

        return scores

    def calculate_attention(self, bags, lens):
        embeddings = self.encoder(bags)
        return self._masked_attention_scores(embeddings, lens)

    def _masked_attention_scores(self, embeddings, lens):
        """Calculates attention scores for all bags.
        Returns:
            A tensor containing torch.concat([torch.rand(64, 256), torch.rand(64, 23)], -1)
             *  The attention score of instance i of bag j if i < len[j]
             *  0 otherwise
        """
        bs, bag_size = embeddings.shape[0], embeddings.shape[1]
        attention_scores = self.attention(embeddings)

        # a tensor containing a row [0, ..., bag_size-1] for each batch instance
        idx = torch.arange(bag_size).repeat(bs, 1).to(attention_scores.device)

        # False for every instance of bag i with index(instance) >= lens[i]
        attention_mask = (idx < lens.unsqueeze(-1)).unsqueeze(-1)

        masked_attention = torch.where(
            attention_mask, attention_scores, torch.full_like(attention_scores, -1e10)
        )
        return torch.softmax(masked_attention, dim=1)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = self.encoder.to(device)
        self.attention = self.attention.to(device)
        self.head = self.head.to(device)

    def plot(*args, **kwargs):
        pass

# -----------------------------------------------------------------------------

def Attention(n_in: int, n_latent: Optional[int] = None) -> nn.Module:
    """A network calculating an embedding's importance weight."""
    n_latent = n_latent or (n_in + 1) // 2

    return nn.Sequential(nn.Linear(n_in, n_latent), nn.Tanh(), nn.Linear(n_latent, 1))