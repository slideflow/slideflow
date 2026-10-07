import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional

# -----------------------------------------------------------------------------

class NNMIL(nn.Module):
    """nnMIL aggregator with random feature-subset attention.

    Gated attention MIL in which the first attention layer only sees a subset
    of the input feature dimensions. During training, a new random subset of
    ``hidden_dim`` dimensions is drawn for every forward pass. At inference,
    predictions are averaged over a fixed set of overlapping feature subsets
    (a sliding window over one fixed permutation of the feature dimensions).
    Attention pooling and the classifier always use all feature dimensions.

    Reference: Luo X, Xiang J, Ji Y, Li R. nnMIL: a generalizable multiple
    instance learning framework for computational pathology. Nature Biomedical
    Engineering (2026). https://arxiv.org/abs/2511.14907

    """

    use_lens = True
    uq_uses_softmax = True

    def __init__(
        self,
        n_feats: int,
        n_out: int,
        hidden_dim: int = 256,
        *,
        dropout_p: float = 0.25,
        stride_divisor: int = 4,
        subset_seed: int = 42,
    ) -> None:
        """Create a new nnMIL model.

        Args:
            n_feats (int): Number of features per instance.
            n_out (int): Number of model outputs.
            hidden_dim (int): Width of the attention network, which is also the
                number of feature dimensions in each subset. Defaults to 256.

        Keyword args:
            dropout_p (float): Dropout on the attention branches. Defaults to 0.25.
            stride_divisor (int): At inference, subsets start every
                ``hidden_dim // stride_divisor`` dimensions. Larger values give
                more overlapping subsets. Defaults to 4.
            subset_seed (int): Seed for the fixed feature permutation used at
                inference. Defaults to 42.

        """
        super().__init__()
        self.n_feats = n_feats
        self.keep = min(hidden_dim, n_feats)
        self.attention_V = nn.Linear(n_feats, hidden_dim)
        self.attention_U = nn.Linear(n_feats, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout_p)
        self.head = nn.Linear(n_feats, n_out)
        self.register_buffer(
            'eval_subsets',
            self._eval_subsets(n_feats, self.keep, stride_divisor, subset_seed)
        )

    @staticmethod
    def _eval_subsets(n_feats, keep, stride_divisor, seed):
        """Overlapping windows over one fixed permutation of the features."""
        if keep >= n_feats:
            return torch.arange(n_feats).unsqueeze(0)
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n_feats, generator=g)
        stride = max(1, keep // stride_divisor)
        starts = list(range(0, n_feats - keep + 1, stride))
        if starts[-1] != n_feats - keep:
            starts.append(n_feats - keep)
        return torch.stack([perm[s:s + keep] for s in starts])

    def _attention_logits(self, bags, lens, idx):
        """Unnormalized attention for each instance, from features ``idx``.

        Padded instances (index >= lens) are set to -inf.
        """
        x = bags.index_select(-1, idx)
        a = torch.tanh(F.linear(x, self.attention_V.weight.index_select(1, idx), self.attention_V.bias))
        b = torch.sigmoid(F.linear(x, self.attention_U.weight.index_select(1, idx), self.attention_U.bias))
        scores = self.attention_w(self.dropout(a) * self.dropout(b))
        if lens is not None:
            pos = torch.arange(bags.shape[1], device=bags.device)
            mask = (pos.unsqueeze(0) < lens.unsqueeze(-1)).unsqueeze(-1)
            scores = scores.masked_fill(~mask, float('-inf'))
        return scores

    def _pool(self, bags, lens, idx):
        attention = torch.softmax(self._attention_logits(bags, lens, idx), dim=1)
        pooled = torch.bmm(attention.transpose(1, 2), bags).squeeze(1)
        return self.head(pooled), attention

    def _subsets(self, device):
        if self.training:
            return torch.randperm(self.n_feats, device=device)[:self.keep].unsqueeze(0)
        return self.eval_subsets

    def forward(
        self,
        bags: torch.Tensor,
        lens: Optional[torch.Tensor] = None,
        *,
        return_attention: bool = False,
        uq: bool = False,
        uq_softmax: bool = True
    ):
        """Predict from a batch of bags.

        Args:
            bags (torch.Tensor): Shape (batch, n_instances, n_feats).
            lens (torch.Tensor, optional): Number of real (unpadded) instances
                in each bag. If None, all instances are used.

        Keyword args:
            return_attention (bool): Also return attention, averaged over the
                feature subsets. Defaults to False.
            uq (bool): Return (mean, std) of the predictions across the
                inference feature subsets. Defaults to False.
            uq_softmax (bool): Apply softmax to each subset's prediction before
                computing the UQ mean and std. Defaults to True.

        """
        assert bags.ndim == 3
        if lens is not None:
            assert bags.shape[0] == lens.shape[0]
        logits, attention = [], []
        for idx in self._subsets(bags.device):
            out, att = self._pool(bags, lens, idx)
            logits.append(out)
            attention.append(att)
        logits = torch.stack(logits)

        if uq:
            preds = torch.softmax(logits, dim=-1) if uq_softmax else logits
            scores = (preds.mean(0), preds.std(0, unbiased=False))
        else:
            scores = logits.mean(0)

        if return_attention:
            return scores, torch.stack(attention).mean(0)
        return scores

    @classmethod
    def from_rsimage(cls, path: str, n_feats: int = 1536) -> "NNMIL":
        """Load a trained head saved by rs-image (``heads/fold{k}.pt``).

        rs-image names the layers ``V``, ``U``, ``w``, ``cls`` and the fixed
        inference subsets ``chunks``; the architecture is the same as this
        class. Returns the model in eval mode. Its output is the raw logit;
        rs-image z-scores it with the fold mean/SD in ``model.json``.

        """
        state = torch.load(path, map_location='cpu', weights_only=True)
        n_out, hidden_dim = state['cls.weight'].shape[0], state['V.weight'].shape[0]
        model = cls(n_feats, n_out, hidden_dim=hidden_dim)
        rename = {'V': 'attention_V', 'U': 'attention_U', 'w': 'attention_w', 'cls': 'head', 'chunks': 'eval_subsets'}
        model.load_state_dict({rename[k.split('.', 1)[0]] + k[len(k.split('.', 1)[0]):]: v for k, v in state.items()}, strict=True)
        return model.eval()

    def calculate_attention(self, bags, lens=None, *, apply_softmax=None):
        """Attention for each instance, averaged over the feature subsets.

        With ``apply_softmax=False``, returns the averaged unnormalized scores.
        """
        if apply_softmax is None:
            apply_softmax = bags.shape[1] > 1
        scores = []
        for idx in self._subsets(bags.device):
            s = self._attention_logits(bags, lens, idx)
            scores.append(torch.softmax(s, dim=1) if apply_softmax else s)
        return torch.stack(scores).mean(0)
