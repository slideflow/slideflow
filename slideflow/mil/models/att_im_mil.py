import torch
from torch import nn
from typing import Optional, List

from slideflow import log
from slideflow.model.torch_utils import get_device

# -----------------------------------------------------------------------------

class Attention_IM_MIL(nn.Module):
    """Attention-based multiple instance learning model with intermediate fusioning.

    Implementation from: https://github.com/KatherLab/marugoto

    """

    use_lens = True

    def __init__(
        self,
        n_feats: int,
        n_out: int,
        z_dim: int = 256,
        z_dim_clinical: int = 35,
        clinical_num_features: int = 0,
        *,
        dropout_p: float = 0.5,
        encoder: Optional[nn.Module] = None,
        clinical_encoder: Optional[nn.Module] = None,
        attention: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        attention_gate: float = 0,
        temperature: float = 1.
    ) -> None:
        """Create a new attention MIL model with intermediate fusioning of clinical data by using the Kronecker product.
        Args:
            n_feats (int):  The number of features each bag instance has.
            n_out (int):  The number of output layers of the model.
            z_dim (int):  The dimensionality of the latent space. Defaults to 256.
            z_dim_clinical (int):  The dimensionality of the latent space of the clinical features. Defaults to 35. Decent initial option is to pick the same size as the number of clinical features.
            clinical_num_features (int): The number of features which correspond to the clinical data in the bag instances (Assumed they are concatenated to the end)

        Keyword args:
            dropout_p (float):  The dropout probability. Defaults to 0.5.
            encoder (nn.Module, optional):  A network transforming bag instances into feature vectors.
                If None, a single-layer network with a ReLU activation is used.
            attention (nn.Module, optional):  A network calculating an embedding's importance weight.
                If None, a single-layer network with a tanh activation is used.
            head (nn.Module, optional):  A network calculating the final prediction from the weighted
                embeddings. If None, a single-layer network batch norm and dropout is used.
            temperature (float): Softmax temperature. Defaults to 1.
            attention_gate (float): Gate predictions prior to attention softmax based on this percentile.
                Defaults to 0 (disabled).

        """
        super().__init__()
        self.encoder = encoder or nn.Sequential(nn.Linear(n_feats - clinical_num_features, z_dim), nn.ReLU())
        self.attention = attention or Attention(z_dim)
        self.head = head or nn.Sequential(
            nn.Flatten(), nn.BatchNorm1d(z_dim * z_dim_clinical), nn.Dropout(dropout_p), nn.Linear(z_dim * z_dim_clinical, n_out)
        )
        self.clinical_encoder = clinical_encoder or nn.Sequential(nn.Linear(clinical_num_features, z_dim_clinical), nn.ReLU())
        self.clinical_num_features = clinical_num_features
        self._neg_inf = torch.tensor(-torch.inf)
        self.attention_gate = attention_gate
        self.temperature = temperature
        if temperature != 1.:
            log.debug("Using attention softmax temperature: {}".format(temperature))
        if attention_gate:
            log.debug("Using attention gate: {} percentile".format(attention_gate))

    def forward(self, bags, lens, *, return_attention=False, uq=False, uq_softmax=True):
        assert bags.ndim == 3
        assert bags.shape[0] == lens.shape[0]

        # Split bags into clinical and non-clinical features
        non_clinical_features = bags[..., :-self.clinical_num_features]
        clinical_features = bags[..., -self.clinical_num_features:]

        embeddings = self.encoder(non_clinical_features)

        masked_attention_scores = self._masked_attention_scores(embeddings, lens, apply_softmax=False)

        if self.attention_gate and bags.shape[1] > 1:

            # Attention threshold (75th percentile). Shape = (batch, )
            attention_threshold = torch.quantile(masked_attention_scores, q=self.attention_gate, dim=1, keepdim=True)

            # Indices of high-attention bags (above threshold).
            high_attention_mask = (masked_attention_scores > attention_threshold)[:, :]

            # Weighted embeddings from only high-attention bags.
            masked_attention_scores = torch.where(
                high_attention_mask, masked_attention_scores, torch.full_like(masked_attention_scores, self._neg_inf)
            )

        # Softmax attention. Shape = (batch, n_bags, 1)
        softmax_attention_scores = torch.softmax(masked_attention_scores / self.temperature, dim=1)

        # Weighted embeddings (attention * embeddings). Shape = (batch, n_bags, n_feats)
        weighted_embeddings = (softmax_attention_scores * embeddings)

        # Sum of weighted embeddings. (batch, n_bags, n_feats)->(batch, 1, n_feats)
        weighted_embeddings_sum = weighted_embeddings.sum(-2)

        # Embedding of clinical features
        clinical_features_mean = clinical_features.mean(-2)  # Average clinical features across instances as they are same for every bag (batch, n_bags, n_feats)->(batch, 1, n_feats)
        clinical_embeddings = self.clinical_encoder(clinical_features_mean)
        

        # Merging clinical and image features
        
        # Concatenate across the feature dimension: (batch, 1, z_dim),(batch, 1, z_dim_clinical)->(batch, 1, z_dim+z_dim_clinical)
        #final_embeddings = torch.cat([weighted_embeddings_sum, clinical_embeddings], dim=-1)
        
        # Kronecker product across the feature dimension of squeezed embeddings: (batch, z_dim),(batch, z_dim_clinical)->(batch, z_dim*z_dim_clinical)
        final_embeddings = torch.einsum('bi,bj->bij', weighted_embeddings_sum.squeeze(1), clinical_embeddings.squeeze(1)).reshape(weighted_embeddings_sum.shape[0], -1) 

        if uq:
            # Expand embeddings 30-fold into second dimension.
            flat = self.head[0](final_embeddings)
            norm = self.head[1](flat)
            expanded = norm.unsqueeze(1).expand(-1, 30, -1)

            # Enable dropout and calculate predictions.
            _prior_status = self.training
            self.head[2].train()
            post_dropout = self.head[2](expanded)
            expanded_preds = self.head[3](post_dropout)
            if uq_softmax:
                expanded_preds = torch.softmax(expanded_preds, dim=2)
            self.train(_prior_status)

            # Average scores across 30 dropout replicates.
            pred_stds = torch.std(expanded_preds, dim=1)
            pred_means = expanded_preds.mean(axis=1)
            scores = (pred_means, pred_stds)

        else:
            scores = self.head(final_embeddings)

        if return_attention and bags.shape[1] > 1:
            return scores, softmax_attention_scores
        elif return_attention:
            return scores, masked_attention_scores
        else:
            return scores

    def calculate_attention(self, bags, lens, *, apply_softmax=None):
        """Calculate attention scores for all bags."""
        if apply_softmax is None and bags.shape[1] > 1:
            apply_softmax = True
        embeddings = self.encoder(bags)
        return self._masked_attention_scores(
            embeddings, lens, apply_softmax=apply_softmax
        )

    def get_last_layer_activations(self, bags, lens):
        assert bags.ndim == 3
        assert bags.shape[0] == lens.shape[0]
        embeddings = self.encoder(bags)
        masked_attention_scores = self._masked_attention_scores(embeddings, lens)
        weighted_embedding_sums = (masked_attention_scores * embeddings).sum(-2)
        return weighted_embedding_sums

    def _masked_attention_scores(self, embeddings, lens, *, apply_softmax=True):
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
            attention_mask, attention_scores, torch.full_like(attention_scores, self._neg_inf)
        )
        if apply_softmax:
            return torch.softmax(masked_attention, dim=1)
        else:
            return masked_attention

    def relocate(self):
        """Move model to GPU. Required for FastAI compatibility."""
        device = get_device()
        self.to(device)
        self._neg_inf = self._neg_inf.to(device)

    def plot(*args, **kwargs):
        pass

# -----------------------------------------------------------------------------

def Attention(n_in: int, n_latent: Optional[int] = None) -> nn.Module:
    """A network calculating an embedding's importance weight."""
    # Note: softmax not being applied here, as it will be applied later,
    # after masking out the padding.
    if n_latent == 0:
        return nn.Linear(n_in, 1)
    else:
        n_latent = n_latent or (n_in + 1) // 2
        return nn.Sequential(
            nn.Linear(n_in, n_latent),
            nn.Tanh(),
            nn.Linear(n_latent, 1)
        )