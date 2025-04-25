import torch
from torch import nn
from typing import Optional, List, Union

from slideflow import log
from slideflow.model.torch_utils import get_device


# -----------------------------------------------------------------------------

class Attention_MIL(nn.Module):
    """Attention-based multiple instance learning model.

    Implementation from: https://github.com/KatherLab/marugoto

    """

    use_lens = True

    def __init__(
        self,
        n_feats: int,
        n_out: int,
        z_dim: int = 256,
        *,
        dropout_p: float = 0.5,
        encoder: Optional[nn.Module] = None,
        attention: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        attention_gate: float = 0,
        temperature: float = 1.
    ) -> None:
        """Create a new attention MIL model.
        Args:
            n_feats (int):  The number of features each bag instance has.
            n_out (int):  The number of output layers of the model.
            z_dim (int):  The dimensionality of the latent space. Defaults to 256.

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
        self.encoder = encoder or nn.Sequential(nn.Linear(n_feats, z_dim), nn.ReLU())
        self.attention = attention or Attention(z_dim)
        self.head = head or nn.Sequential(
            nn.Flatten(), nn.BatchNorm1d(z_dim), nn.Dropout(dropout_p), nn.Linear(z_dim, n_out)
        )
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

        embeddings = self.encoder(bags)

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

        # Sum of weighted embeddings.
        weighted_embeddings_sum = weighted_embeddings.sum(-2)

        if uq:
            # Expand embeddings 30-fold into second dimension.
            flat = self.head[0](weighted_embeddings_sum)
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
            scores = self.head(weighted_embeddings_sum)

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

# -----------------------------------------------------------------------------

class MultiModal_Attention_MIL(nn.Module):
    """Attention-based MIL model for multiple input feature spaces.

    Used for multi-magnification MIL. Differs from Attention_MIL in that it
    takes multiple bags as input, one for each magnification.

    """

    multimodal = True
    use_lens = True

    def __init__(
        self,
        n_feats: List[int],
        n_out: int,
        z_dim: int = 512,
        dropout_p: float = 0.3
    ) -> None:
        super().__init__()
        self.n_input = len(n_feats)
        self._z_dim = z_dim
        self._dropout_p = dropout_p
        self._n_out = n_out
        for i in range(self.n_input):
            setattr(self, f'encoder_{i}', nn.Sequential(nn.Linear(n_feats[i], z_dim), nn.ReLU()))
            setattr(self, f'attention_{i}', Attention(z_dim, n_latent=0))  # Simple, single-layer attention
            setattr(self, f'prehead_{i}', nn.Sequential(nn.Flatten(),
                                                        nn.BatchNorm1d(z_dim),
                                                        nn.ReLU(),
                                                        nn.Dropout(dropout_p)))

        # Concatenate the weighted sums of embeddings from each magnification
        # into a single vector, then pass it through a linear layer.
        self.head = nn.Sequential(
            nn.Linear(z_dim * self.n_input, z_dim),
            nn.LayerNorm(z_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(z_dim, n_out)
        )
        self._neg_inf = torch.tensor(-torch.inf)

    def forward(self, *bags_and_lens):
        """Return predictions using all bags and magnifications.

        Input should be a list of tuples, with each tuple containing a bag and
        lens tensors corresponding to a single magnification level. The length
        of the list should be equal to the number of magnification levels.

        """
        self._verify_input(*bags_and_lens)
        bags, lenses = zip(*bags_and_lens)

        embeddings = self._calculate_embeddings(bags)
        masked_attention_scores = self._all_masked_attention(embeddings, lenses)

        weighted_embeddings = self._calculate_weighted_embeddings(masked_attention_scores, embeddings)
        merged_embeddings = torch.cat(weighted_embeddings, dim=1)

        output = self.head(merged_embeddings)
        return output

    def calculate_attention(self, *bags_and_lens, apply_softmax=True):
        """Calculate attention scores for all bags and magnifications."""
        self._verify_input(*bags_and_lens)
        bags, lenses = zip(*bags_and_lens)

        # Convert bags into embeddings.
        embeddings = self._calculate_embeddings(bags)

        # Calculate masked attention scores from the embeddings.
        masked_attention_scores = self._all_masked_attention(embeddings, lenses, apply_softmax=apply_softmax)

        return masked_attention_scores

    # --- Private methods ------------------------------------------------------

    def _verify_input(self, *bags_and_lens):
        """Verify that the input is valid."""
        if len(bags_and_lens) != self.n_input:
            raise ValueError(
                f'Expected {self.n_input} inputs (tuples of bags and lens), got '
                f'{len(bags_and_lens)}'
            )
        for i in range(self.n_input):
            bags, lens = bags_and_lens[i]
            if bags.ndim != 3:
                raise ValueError(f'Bag tensor {i} has {bags.ndim} dimensions, expected 3')
            if bags.shape[0] != lens.shape[0]:
                raise ValueError(
                    f'Bag tensor {i} has {bags.shape[0]} bags, but lens tensor has '
                    f'{lens.shape[0]} entries'
                )

    def _calculate_weighted_embeddings(self, masked_attention_scores, embeddings):
        return [
            getattr(self, f'prehead_{i}')(torch.sum(mas * emb, dim=1))
            for i, (mas, emb) in enumerate(zip(masked_attention_scores, embeddings))
        ]

    def _calculate_embeddings(self, bags):
        """Calculate embeddings for all magnifications."""
        return [
            getattr(self, f'encoder_{i}')(bags[i])
            for i in range(self.n_input)
        ]

    def _all_masked_attention(self, embeddings, lenses, *, apply_softmax=True):
        """Calculate masked attention scores for all magnification levels."""
        return [
            self._masked_attention_scores(embeddings[i], lenses[i], i, apply_softmax=apply_softmax)
            for i in range(self.n_input)
        ]

    def _masked_attention_scores(self, embeddings, lens, mag_index, *, apply_softmax=True):
        """Calculate masked attention scores at the given magnification.

        Returns:
            A tensor containing torch.concat([torch.rand(64, 256), torch.rand(64, 23)], -1)
             *  The attention score of instance i of bag j if i < len[j]
             *  0 otherwise
        """
        bs, bag_size = embeddings.shape[0], embeddings.shape[1]
        attention_scores = getattr(self, f'attention_{mag_index}')(embeddings)

        # a tensor containing a row [0, ..., bag_size-1] for each batch instance
        idx = torch.arange(bag_size).repeat(bs, 1).to(attention_scores.device)

        # False for every instance of bag i with index(instance) >= lens[i]
        attention_mask = (idx < lens.unsqueeze(-1)).unsqueeze(-1)

        masked_attention = torch.where(attention_mask, attention_scores, self._neg_inf)
        if apply_softmax:
            return torch.softmax(masked_attention, dim=1)
        else:
            return masked_attention

    # --- FastAI compatibility -------------------------------------------------

    def relocate(self):
        """Move model to GPU. Required for FastAI compatibility."""
        device = get_device()
        self.to(device)
        self._neg_inf = self._neg_inf.to(device)

    def plot(*args, **kwargs):
        """Override to disable FastAI plotting."""
        pass


class UQ_MultiModal_Attention_MIL(MultiModal_Attention_MIL):
    """Variant of the MultiModal attention-MIL model with uncertainty-weighted fusion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.head = nn.Sequential(
            nn.Linear(self._z_dim, self._z_dim),
            nn.LayerNorm(self._z_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self._z_dim, self._n_out)
        )
        self.uq_dropout = nn.Dropout(self._dropout_p)

    def forward(self, *bags_and_lens):
        """Return predictions using all bags and magnifications.

        Input should be a list of tuples, with each tuple containing a bag and
        lens tensors corresponding to a single magnification level. The length
        of the list should be equal to the number of magnification levels.

        """
        self._verify_input(*bags_and_lens)
        bags, lenses = zip(*bags_and_lens)

        embeddings = self._calculate_embeddings(bags)
        masked_attention_scores = self._all_masked_attention(embeddings, lenses)

        merged_embeddings = self._merge_uncertainty_weighted_embeddings(embeddings, masked_attention_scores)

        output = self.head(merged_embeddings)
        return output

    def _merge_uncertainty_weighted_embeddings(self, embeddings, masked_attention_scores):
        weighted_embeddings = self._calculate_weighted_embeddings(masked_attention_scores, embeddings)
        expanded_embeddings = [emb.unsqueeze(1).expand(-1, 30, -1) for emb in weighted_embeddings]
        mode_uncertainty = self._calculate_mode_uncertainty(expanded_embeddings)

        # Weight the embeddings from each magnification by their uncertainty.
        stacked_uncertainty = torch.stack(mode_uncertainty, dim=1)
        uncertainty_weights = 1 - torch.softmax(stacked_uncertainty, dim=1)

        final_weighted_embeddings = self._merge_weighted_embeddings(
            masked_attention_scores, embeddings, uncertainty_weights
        )
        return torch.sum(final_weighted_embeddings, dim=1)

    def _calculate_mode_uncertainty(self, expanded_embeddings):
        mode_uncertainty = []
        for i in range(self.n_input):
            # Enforce dropout.
            _prior_status = self.training
            self.uq_dropout.train()
            dropout_expanded = self.uq_dropout(expanded_embeddings[i])
            self.train(_prior_status)

            # Concatenate the expanded embedding sums with the original
            # embedding sums.
            all_embeddings = [
                    (expanded_embeddings[j] if j!=i else dropout_expanded) * 0.5
                    for j in range(self.n_input)
                ]
            all_embeddings = torch.sum(torch.stack(all_embeddings, dim=2), dim=2)

            # Pass the concatenated embeddings through a final linear layer.
            expanded_scores = self.head(all_embeddings)

            # Average the scores across the 30 dropout samples.
            score_stds = torch.std(expanded_scores, dim=1)
            avg_by_batch = score_stds.mean(axis=1)

            mode_uncertainty.append(avg_by_batch)

        return mode_uncertainty

    def _merge_weighted_embeddings(self, masked_attention_scores, embeddings, uncertainty_weights):
        return torch.stack([
            getattr(self, f'prehead_{i}')(torch.sum(mas * emb, dim=1)) * uncertainty_weights[:, i].unsqueeze(-1)
            for i, (mas, emb) in enumerate(zip(masked_attention_scores, embeddings))
        ], dim=1)

# -----------------------------------------------------------------------------

class MultiModal_Mixed_Attention_MIL(nn.Module):
    """Attention-based MIL model for mixed modality inputs with potential missing modalities."""

    use_lens = True

    def __init__(
        self,
        n_feats: List[int],
        n_out: int,
        z_dim: int = 256, # latent space dimension
        *,
        hidden_dim: Optional[Union[List[int], int]] = None, # same length as n_feats if None
        n_layers: Optional[Union[List[int], int]] = None, 
        dropout_p: float = 0.5,
        temperature: float = 1.
    ) -> None:
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = [z_dim] * len(n_feats)
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim] * len(n_feats)
        if n_layers is None: # default is 1 layer
            n_layers = [1] * len(n_feats)
        if isinstance(n_layers, int):
            n_layers = [n_layers] * len(n_feats)

        # check if hidden_dim is same length as n_feats
        if len(hidden_dim) != len(n_feats) or len(n_layers) != len(n_feats):
            raise ValueError("hidden_dim and n_layers must be the same length as n_feats")
        
        self.n_modalities = len(n_feats)
        self.z_dim = z_dim
        self.n_feats = n_feats  # Save feature lengths for each modality
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Create an encoder for each modality that projects to shared z_dim space
        self.encoders = nn.ModuleList()
        self.fc_mus = nn.ModuleList()
        self.fc_vars = nn.ModuleList()
        
        for i, nf in enumerate(n_feats):
            # Encoder layers
            layers = []
            # First layer (input to hidden)
            layers.extend([
                nn.Linear(in_features=nf, out_features=hidden_dim[i]),
                nn.BatchNorm1d(hidden_dim[i]),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
            
            # Middle layers (hidden to hidden)
            for _ in range(n_layers[i] - 1):
                layers.extend([
                    nn.Linear(in_features=hidden_dim[i], out_features=hidden_dim[i]),
                    nn.BatchNorm1d(hidden_dim[i]),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1)
                ])
            
            self.encoders.append(nn.Sequential(*layers))
            
            # Mean and variance projections
            self.fc_mus.append(nn.Linear(hidden_dim[i], z_dim))
            self.fc_vars.append(nn.Linear(hidden_dim[i], z_dim))
        
        # Create decoders that map the latent space into for each modality space
        self.decoders = nn.ModuleList()
        for i, nf in enumerate(n_feats):
            decoder_layers = []
            # First layer (latent to hidden)
            decoder_layers.extend([
                nn.Linear(in_features=z_dim, out_features=hidden_dim[i]),
                nn.BatchNorm1d(hidden_dim[i]),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
            
            # Middle layers (hidden to hidden)
            for _ in range(n_layers[i] - 1):
                decoder_layers.extend([
                    nn.Linear(in_features=hidden_dim[i], out_features=hidden_dim[i]),
                    nn.BatchNorm1d(hidden_dim[i]),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1)
                ])
            
            # Final projection back to original feature space
            decoder_layers.append(nn.Linear(in_features=hidden_dim[i], out_features=nf))
            
            self.decoders.append(nn.Sequential(*decoder_layers))

        # Single attention mechanism for the shared embedding space
        self.attention = Attention(z_dim)
        
        # Final prediction head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(z_dim),
            nn.Dropout(dropout_p),
            nn.Linear(z_dim, n_out)
        )
        
        self._neg_inf = torch.tensor(-torch.inf)
        self.temperature = temperature

    def forward(self, *inputs, attention=False, decode=True):
        """Forward pass through the network.
        
        Args:
            *inputs: Variable number of arguments where:
                - The first N-1 arguments are tensors for each modality
                - The last argument is the modality mask
        """
        modalities = inputs[:-1]  # All but last input are modality tensors
        modality_mask = inputs[-1]  # Last input is the mask
        
        # Encode each modality to means and variances
        mus, vars = self.encode_all(modalities)

        # Stack encoded vectors for attention
        # Shape: (batch_size, n_modalities, z_dim)
        stacked_embeddings = torch.stack(mus, dim=1)
        
        # Calculate masked attention scores
        masked_attention = self._masked_attention_scores(
            stacked_embeddings, 
            modality_mask,
            apply_softmax=True
        )
        
        # Apply attention weights to embeddings
        weighted_embeddings = (masked_attention * stacked_embeddings)
        
        # Sum across modalities
        pooled_embeddings = weighted_embeddings.sum(dim=1)  # Shape: (batch_size, z_dim)
        
        # Final prediction
        predictions = self.head(pooled_embeddings)

        if attention:
            res = predictions, masked_attention
        else:
            res = predictions

        if not self.training:
            return res

        if decode:
            # Get latent representations through reparameterization
            zs = self.reparameterize(mus, vars)
            # Decode latent representations back to modality space
            res = res, self.decode_all(zs), modalities, modality_mask

        return res

    def encode_all(self, modalities):
        """Encode all modalities to their means and variances in latent space.
        
        Args:
            modalities: List of tensors, one for each modality
            
        Returns:
            mus: List of mean tensors, one for each modality
            vars: List of variance tensors, one for each modality
        """
        mus, vars = [], []
        for i, modality in enumerate(modalities):
            h = self.encoders[i](modality)
            mu = self.fc_mus[i](h)
            var = self.fc_vars[i](h)
            mus.append(mu)
            vars.append(var)
        return mus, vars

    def decode_all(self, z):
        """Decode latent vectors back to all modality spaces.
        
        Args:
            z: Tensor of shape (batch_size, z_dim) or list of such tensors
            
        Returns:
            reconstructions: List of tensors, where reconstructions[i] contains 
                reconstructions into modality i's space.
                Shape is (batch_size, n_modalities, modality_i_dim)
        """
        if isinstance(z, list):
            reconstructions = [ 
                torch.stack([self.decoders[i](z_i) for z_i in z], dim=1) 
                for i in range(self.n_modalities)
            ] # (batch_size, n_modalities, modality_i_dim)

            return reconstructions

    def reparameterize(self, mus, logvars):
        """Reparameterize for each modality using the respective mu and logvar.
        
        Args:
            mus: List of mean tensors, one for each modality
            logvars: List of log variance tensors, one for each modality
            
        Returns:
            zs: List of sampled latent vectors, one for each modality
        """
        zs = []
        for mu, logvar in zip(mus, logvars):
            std = logvar.mul(0.5).exp_()
            eps = torch.normal(mean=0, std=1, size=std.size(), device=mu.device)
            z = eps.mul(std).add_(mu)
            zs.append(z)
        return zs

    def _masked_attention_scores(self, embeddings, modality_mask, *, apply_softmax=True):
        """Calculate masked attention scores.
        
        Args:
            embeddings: Tensor of shape (batch_size, n_modalities, z_dim)
            modality_mask: Tensor of shape (batch_size, n_modalities)
            apply_softmax: Whether to apply softmax to the masked attention scores
        
        Returns:
            Tensor of shape (batch_size, n_modalities, 1) containing attention scores
        """
        attention_scores = self.attention(embeddings)  # Shape: (batch_size, n_modalities, 1)
        
        # Mask out attention for missing modalities
        masked_attention = torch.where(
            modality_mask.unsqueeze(-1),
            attention_scores,
            torch.full_like(attention_scores, self._neg_inf)
        )
        
        if apply_softmax:
            return torch.softmax(masked_attention / self.temperature, dim=1)
        return masked_attention

    def relocate(self):
        device = get_device()
        self.to(device)
        self._neg_inf = self._neg_inf.to(device)

    def plot(*args, **kwargs):
        pass

