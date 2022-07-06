import torch.nn as nn


class Classifier(nn.Module):
	def __init__(self, d_model=80, n_spks=600, dropout=0.1):
		super().__init__()
		# Project the dimension of features from that of input into d_model.
		self.prenet = nn.Linear(40, d_model)
		# TODO:
		#   Change Transformer to Conformer.
		#   https://arxiv.org/abs/2005.08100
		self.encoder_layer = nn.TransformerEncoderLayer(
			d_model=d_model, dim_feedforward=256, nhead=2
		)
		# self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

		# Project the the dimension of features from d_model into speaker nums.
		self.pred_layer = nn.Sequential(
			nn.Linear(d_model, d_model),
			nn.ReLU(),
			nn.Linear(d_model, n_spks),
		)

	def forward(self, mels):
		"""
		args:
			mels: (batch size, length, 40)
		return:
			out: (batch size, n_spks)
		"""
		# mels: (batch size, 128, 40)
		out = self.prenet(mels)
		# out: (batch size, 128, 80)
		out = out.permute(1, 0, 2)
		# out: (128, batch size, 80)
		# The encoder layer expect features in the shape of (length, batch size, d_model).
		out = self.encoder_layer(out)
		# out: (128, batch size, 80)
		# is still a size of mel spectrum
		out = out.transpose(0, 1)
		#out: (batch size, 128, 80)

		# mean pooling of a spectrum length
		stats = out.mean(dim=1) 
		#stats: (batch size, 80)

		out = self.pred_layer(stats)
		# out: (batch, n_spks)
		# each mel specturm retrun a speaker

		return out