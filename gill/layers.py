import torch
from torch import nn


class TextFcLayer(nn.Module):
  """Layers used in mapping text embeddings to visual outputs."""

  def __init__(self, in_dim: int, out_dim: int, num_input_tokens: int = 1, num_output_tokens: int = 1, mode: str = 'linear'):
    super().__init__()

    self.num_input_tokens = num_input_tokens
    self.num_output_tokens = num_output_tokens
    self.mode = mode

    if mode == 'linear':
      self.model = nn.Linear(in_dim, out_dim)
    elif mode == 'gill_mapper':  # TODO(jykoh): Rename to GILLMapper
      hidden_dim = 512
      self.fc = nn.Linear(in_dim, hidden_dim)
      self.tfm = nn.Transformer(batch_first=True, norm_first=True,
                                d_model=hidden_dim, num_encoder_layers=4, num_decoder_layers=4,
                                dim_feedforward=hidden_dim * 4, dropout=0.0, nhead=4)
      self.model = nn.Linear(hidden_dim, out_dim)
      self.query_embs = nn.Parameter(torch.randn(1, num_output_tokens, hidden_dim))
    else:
      raise NotImplementedError(mode)

  def forward(self, x: torch.Tensor, input_embs: torch.Tensor) -> torch.Tensor:
    outputs = None
    
    if self.mode == 'gill_mapper':
      x = x + input_embs

    if isinstance(self.model, nn.ModuleList):
      assert len(self.model) == x.shape[1] == self.num_input_tokens, (len(self.model), x.shape, self.num_input_tokens)
      outputs = []
      for i in range(self.num_input_tokens):
        outputs.append(self.model[i](x[:, i, :]))  # (N, D)
      outputs = torch.stack(outputs, dim=1)  # (N, T, D)
    else:
      if self.mode == 'gill_mapper':
        x = self.fc(x)
        x = self.tfm(x, self.query_embs.repeat(x.shape[0], 1, 1))
      outputs = self.model(x)

      if outputs.shape[1] != self.num_output_tokens and self.mode == 'linear':
        if self.mode == 'linear':
          outputs = outputs[:, :self.num_output_tokens, :]
        else:
          raise NotImplementedError
    
    assert outputs.shape[1] == 1 or (outputs.shape[1] * outputs.shape[2] == self.num_output_tokens * 768), (outputs.shape, self.num_output_tokens)
    return outputs  # (N, T, D)

