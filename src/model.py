# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import src.utils as utils
from src.constants import ACTIVATION_FUNCS

def _get_weight_init(options):
    return getattr(options, "weight_init", "default").lower()

def _get_init_gain(options):
    return float(getattr(options, "init_gain", 1.0))

def _init_linear_weight(weight, init_type, gain):
    if init_type == "default":
        return
    if init_type == "kaiming_uniform":
        nn.init.kaiming_uniform_(weight, a=0.0, nonlinearity="relu")
    elif init_type == "kaiming_normal":
        nn.init.kaiming_normal_(weight, a=0.0, nonlinearity="relu")
    else:
        raise ValueError(f"Unknown weight_init: {init_type}")

def _init_rnn_weight(weight, init_type, gain):
    if init_type == "default":
        return
    if init_type == "kaiming_uniform":
        nn.init.kaiming_uniform_(weight, a=0.0, nonlinearity="relu")
    elif init_type == "kaiming_normal":
        nn.init.kaiming_normal_(weight, a=0.0, nonlinearity="relu")
    else:
        raise ValueError(f"Unknown weight_init: {init_type}")


class DualInputRNN(nn.Module):
    """
    Single-layer dual-input RNN.

    Update rule:
        h_t = act(
            x1_t @ weight_ih_l0.T + bias_ih_l0
          + x2_t @ weight_jh_l0.T + bias_jh_l0
          + h_{t-1} @ weight_hh_l0.T + bias_hh_l0
        )

    Inputs:
        input1: [B, L, I1] if batch_first=True else [L, B, I1]
        input2: [B, L, I2] if batch_first=True else [L, B, I2]
        hx:     [1, B, H] (optional)

    Returns:
        output: [B, L, H] if batch_first=True else [L, B, H]
        hn:     [1, B, H]
    """
    def __init__(
        self,
        input1_size,
        input2_size,
        hidden_size,
        nonlinearity="tanh",
        bias=False,
        batch_first=True,
    ):
        super().__init__()
        self.input1_size = input1_size
        self.input2_size = input2_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.bias = bias
        self.nonlinearity = nonlinearity

        if nonlinearity not in ("tanh", "relu"):
            raise ValueError("nonlinearity must be 'tanh' or 'relu'")

        # Match nn.RNN-style naming where possible
        self.weight_ih_l0 = nn.Parameter(torch.empty(hidden_size, input1_size))
        self.weight_jh_l0 = nn.Parameter(torch.empty(hidden_size, input2_size))  # second input
        self.weight_hh_l0 = nn.Parameter(torch.empty(hidden_size, hidden_size))

        if bias:
            self.bias_ih_l0 = nn.Parameter(torch.empty(hidden_size))
            self.bias_jh_l0 = nn.Parameter(torch.empty(hidden_size))
            self.bias_hh_l0 = nn.Parameter(torch.empty(hidden_size))
        else:
            self.register_parameter("bias_ih_l0", None)
            self.register_parameter("bias_jh_l0", None)
            self.register_parameter("bias_hh_l0", None)

        self.reset_parameters()

    def reset_parameters(self):
        # Same scale convention as torch RNN default
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            nn.init.uniform_(p, -stdv, stdv)

    def _act(self, x):
        return torch.tanh(x) if self.nonlinearity == "tanh" else F.relu(x)

    def forward(self, input1, input2, hx=None):
        if input1.dim() != 3 or input2.dim() != 3:
            raise ValueError("input1 and input2 must be 3D tensors")
        if input1.shape[:2] != input2.shape[:2]:
            raise ValueError("input1 and input2 must share batch and sequence dims")

        if self.batch_first:
            B, L, _ = input1.shape
            x1 = input1
            x2 = input2
        else:
            L, B, _ = input1.shape
            x1 = input1.transpose(0, 1)  # -> [B, L, I1]
            x2 = input2.transpose(0, 1)  # -> [B, L, I2]

        if hx is None:
            h_t = torch.zeros(B, self.hidden_size, device=x1.device, dtype=x1.dtype)
        else:
            if hx.shape != (1, B, self.hidden_size):
                raise ValueError(f"hx must have shape [1, {B}, {self.hidden_size}]")
            h_t = hx[0]

        outputs = []
        for t in range(L):
            pre = (
                F.linear(x1[:, t, :], self.weight_ih_l0, self.bias_ih_l0)
                + F.linear(x2[:, t, :], self.weight_jh_l0, self.bias_jh_l0)
                + F.linear(h_t, self.weight_hh_l0, self.bias_hh_l0)
            )
            h_t = self._act(pre)
            outputs.append(h_t)

        output = torch.stack(outputs, dim=1)  # [B, L, H]
        hn = h_t.unsqueeze(0)                 # [1, B, H]

        if not self.batch_first:
            output = output.transpose(0, 1)   # [L, B, H]

        return output, hn


class RNN(torch.nn.Module):
    def __init__(self, options, place_cells):
        super(RNN, self).__init__()
        self.Ng = options.Ng
        self.Np = options.Np
        self.sequence_length = options.sequence_length
        self.weight_decay = options.weight_decay
        self.place_cells = place_cells
        self.loss = options.loss
        self.truncating = options.truncating
        self.use_prev_input = options.use_prev_input

        # Input weights
        self.encoder = torch.nn.Linear(self.Np, self.Ng, bias=False)
        if self.use_prev_input:
            self.RNN = DualInputRNN(
                input1_size=2,
                input2_size=self.Np,
                hidden_size=self.Ng,
                nonlinearity=options.rec_activation,
                bias=False,
                batch_first=True,
            )
        else:
            self.RNN = torch.nn.RNN(
                input_size=2,
                hidden_size=self.Ng,
                nonlinearity=options.rec_activation,
                bias=False,
                batch_first=True,
            )
        # Linear read-out weights
        self.decoder = torch.nn.Linear(self.Ng, self.Np, bias=True)
        self._apply_weight_init(options)

        if options.out_activation == "softmax":
            self.out_activation = torch.nn.Softmax(dim=-1)
        elif options.out_activation == "tanh":
            self.out_activation = torch.nn.Tanh()

    def _apply_weight_init(self, options):
        init_type = _get_weight_init(options)
        gain = _get_init_gain(options)
        _init_linear_weight(self.encoder.weight, init_type, gain)
        _init_rnn_weight(self.RNN.weight_ih_l0, init_type, gain)
        if self.use_prev_input:
            _init_rnn_weight(self.RNN.weight_jh_l0, init_type, gain)
        _init_rnn_weight(self.RNN.weight_hh_l0, init_type, gain)
        _init_linear_weight(self.decoder.weight, init_type, gain)
        if self.decoder.bias is not None and init_type != "default":
            nn.init.zeros_(self.decoder.bias)

    def g(self, inputs, pc_outputs):
        """
        Compute grid cell activations.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            pc_outputs: Batch of place cell activations with shape [batch_size, sequence_length, Np].

        Returns:
            g: Batch of grid cell activations with shape [batch_size, sequence_length, Ng].
        """
        if self.truncating == 0:
            v, p0 = inputs
            init_state = self.encoder(p0)[None]

            if self.use_prev_input:
                pc_inputs = torch.zeros_like(pc_outputs)
                pc_inputs[:, 1:, :] = pc_outputs[:, :-1, :]    
                g, _ = self.RNN(v, pc_inputs, init_state)  # use place cell activity from previous timestep as input to RNN
            else:
                g, _ = self.RNN(v, init_state)
            return g
        
        else:
            total_g = []
            vs, p0 = inputs
            seq_len = vs.size(1)
            h = self.encoder(p0)[None]
            for k in range(0, seq_len, self.truncating):
                v = vs[:, k : min(k + self.truncating, seq_len)]  # bsz, trunc, 2
                g, h = self.RNN(v, h)  # g: bsz, trunc, Ng; h: bsz, 1, Ng (final hidden state to be used in next iter)
                h = h.detach()
                total_g.append(g)
            total_g = torch.cat(total_g, dim=1)
            return total_g  # bsz, seq_len, Ng

    def predict(self, inputs, pc_outputs=None):
        """
        Predict place cell code.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            pc_outputs: Batch of place cell activations with shape [batch_size, sequence_length, Np].
        Returns:
            place_preds: Predicted place cell activations with shape
                [batch_size, sequence_length, Np].
        """
        if self.use_prev_input:
            place_preds = self.decoder(self.g(inputs, pc_outputs=pc_outputs))
        else:
            place_preds = self.decoder(self.g(inputs, pc_outputs=None))

        return place_preds

    def compute_loss(self, inputs, pc_outputs, pos):
        """
        Compute avg. loss and decoding error.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            pc_outputs: Ground truth place cell activations with shape
                [batch_size, sequence_length, Np].
            pos: Ground truth 2d position with shape [batch_size, sequence_length, 2].

        Returns:
            loss: Avg. loss for this training batch.
            err: Avg. decoded position error in cm.
        """
        y = pc_outputs
        if self.use_prev_input:
            preds = self.out_activation(self.predict(inputs, pc_outputs=pc_outputs))
        else:
            preds = self.out_activation(self.predict(inputs, pc_outputs=None))
        if self.loss == "CE":
            loss = -(y * torch.log(preds + 1e-9)).sum(-1).mean()
        elif self.loss == "MSE":
            loss = torch.sum((preds - y) ** 2, -1).mean()

        # Weight regularization
        loss += self.weight_decay * (self.RNN.weight_hh_l0**2).sum()

        # Compute decoding error
        preds = F.softmax(preds, dim=-1)
        pred_pos = self.place_cells.get_nearest_cell_pos(preds)
        err = torch.sqrt(((pos - pred_pos) ** 2).sum(-1)).mean()

        return loss, err


class HierarchicalPCN(nn.Module):
    def __init__(self, options):
        super().__init__()

        self.Wout = nn.Linear(options.Ng, options.Np, bias=False)
        self.mu = nn.Parameter(torch.zeros((options.Ng)))

        # sparse penalty
        self.sparse_z = options.lambda_z_init

        self.out_activation = ACTIVATION_FUNCS[options.out_activation]
        self.loss = options.loss

    def set_sparsity(self, sparsity):
        self.sparse_z = sparsity

    def set_nodes(self, inp):
        # intialize the value nodes
        self.z = self.mu.clone()
        self.x = inp.clone()

        # computing error nodes
        self.update_err_nodes()

    def decode(self, z):
        return self.out_activation(self.Wout(z))

    def update_err_nodes(self):
        self.err_z = self.z - self.mu
        pred_x = self.decode(self.z)
        if isinstance(self.out_activation, utils.Tanh):
            self.err_x = self.x - pred_x
        elif isinstance(self.out_activation, utils.Softmax):
            self.err_x = self.x / (pred_x + 1e-8)
        else:
            self.err_x = self.x / (pred_x + 1e-8) + (1 - self.x) / (1 - pred_x + 1e-8)

    def inference_step(self, inf_lr):
        Wout = self.Wout.weight.clone().detach()
        if isinstance(self.out_activation, utils.Softmax):
            delta = (
                self.err_z
                - (
                    self.out_activation.deriv(self.Wout(self.z))
                    @ self.err_x.unsqueeze(-1)
                ).squeeze(-1)
                @ Wout
            )
        else:
            delta = (
                self.err_z
                - (self.out_activation.deriv(self.Wout(self.z)) * self.err_x) @ Wout
            )
        delta += self.sparse_z * torch.sign(self.z)
        self.z = self.z - inf_lr * delta

    def inference(self, inf_iters, inf_lr, inp):
        self.set_nodes(inp)
        for itr in range(inf_iters):
            with torch.no_grad():
                self.inference_step(inf_lr)
            self.update_err_nodes()

    def get_energy(self):
        """Function to obtain the sum of all layers' squared MSE"""
        if self.loss == "CE":
            obs_loss = F.cross_entropy(self.Wout(self.z), self.x)
        elif self.loss == "BCE":
            obs_loss = F.binary_cross_entropy_with_logits(self.Wout(self.z), self.x)
        else:
            obs_loss = torch.sum(self.err_x**2, -1).mean()
        latent_loss = torch.sum(self.err_z**2, -1).mean()
        energy = obs_loss + latent_loss
        return energy, obs_loss


class TemporalPCN(nn.Module):
    """Multi-layer tPC class, using autograd"""

    def __init__(self, options):
        super(TemporalPCN, self).__init__()
        self.Wr = nn.Linear(options.Ng, options.Ng, bias=False)
        self.Win = nn.Linear(options.Nv, options.Ng, bias=False)
        self.Wout = nn.Linear(options.Ng, options.Np, bias=False)
        self._apply_weight_init(options)

        if options.no_velocity:
            self.Win.weight.data.fill_(0)
            self.Win.weight.requires_grad = False

        self.sparse_z = options.lambda_z
        self.weight_decay = options.weight_decay

        self.out_activation = ACTIVATION_FUNCS[options.out_activation]
        self.rec_activation = ACTIVATION_FUNCS[options.rec_activation]
        self.loss = options.loss

    def _apply_weight_init(self, options):
        init_type = _get_weight_init(options)
        gain = _get_init_gain(options)
        _init_linear_weight(self.Wr.weight, init_type, gain)
        _init_linear_weight(self.Win.weight, init_type, gain)
        _init_linear_weight(self.Wout.weight, init_type, gain)

    def set_nodes(self, v, prev_z, p):
        """Set the initial value of the nodes;

        In particular, we initialize the hiddden state with a forward pass.

        Args:
            v: velocity input at a particular timestep in stimulus
            prev_z: previous hidden state
            p: place cell activity at a particular timestep in stimulus
        """
        self.z = self.g(v, prev_z)
        self.x = p.clone()
        self.update_err_nodes(v, prev_z)

    def update_err_nodes(self, v, prev_z):
        self.err_z = self.z - self.g(v, prev_z)
        pred_x = self.decode(self.z)
        if isinstance(self.out_activation, utils.Tanh):
            self.err_x = self.x - pred_x
        elif isinstance(self.out_activation, utils.Softmax):
            self.err_x = self.x / (pred_x + 1e-9)
        else:
            self.err_x = self.x / (pred_x + 1e-9) - (1 - self.x) / (1 - pred_x + 1e-9)

    def g(self, v, prev_z):
        return self.rec_activation(self.Wr(prev_z) + self.Win(v))

    def decode(self, z):
        return self.out_activation(self.Wout(z))

    def inference_step(self, inf_lr, v, prev_z):
        """Take a single inference step"""
        Wout = self.Wout.weight.detach().clone()  # shape [Np, Ng]
        if isinstance(self.out_activation, utils.Softmax):
            delta = (
                self.err_z
                - (
                    self.out_activation.deriv(self.Wout(self.z))
                    @ self.err_x.unsqueeze(-1)
                ).squeeze(-1)
                @ Wout
            )
        else:
            delta = (
                self.err_z
                - (self.out_activation.deriv(self.Wout(self.z)) * self.err_x) @ Wout
            )
        delta += self.sparse_z * torch.sign(self.z)
        self.z = self.z - inf_lr * delta

    def inference(self, inf_iters, inf_lr, v, prev_z, p):
        """Run inference on the hidden state"""
        self.set_nodes(v, prev_z, p)
        for i in range(inf_iters):
            with torch.no_grad():  # ensures self.z won't have grad when we call backward
                self.inference_step(inf_lr, v, prev_z)
            self.update_err_nodes(v, prev_z)

    def get_energy(self):
        """Returns the average (across batches) energy of the model"""
        if self.loss == "CE":
            obs_loss = F.cross_entropy(self.Wout(self.z), self.x)
        elif self.loss == "BCE":
            obs_loss = F.binary_cross_entropy_with_logits(self.Wout(self.z), self.x)
        else:
            obs_loss = torch.sum(self.err_x**2, -1).mean()
        latent_loss = torch.sum(self.err_z**2, -1).mean()
        energy = obs_loss + latent_loss
        energy += self.weight_decay * (torch.mean(self.Wr.weight**2))

        return energy, obs_loss

# TODO: replace multilayerPCN with HierarchicalPCN in pc_pcn.py
class MultilayerPCN(nn.Module):
    def __init__(self, nodes, nonlin, lamb=0.0, use_bias=False, relu_inf=True):
        super().__init__()
        self.n_layers = len(nodes)
        self.layers = nn.Sequential()
        for l in range(self.n_layers - 1):
            self.layers.add_module(
                f"layer_{l}",
                nn.Linear(
                    in_features=nodes[l],
                    out_features=nodes[l + 1],
                    bias=use_bias,
                ),
            )

        self.mem_dim = nodes[0]
        self.memory = nn.Parameter(torch.zeros((nodes[0],)))
        self.relu_inf = relu_inf

        if nonlin == "tanh":
            nonlin = utils.Tanh()
        elif nonlin == "ReLU":
            nonlin = utils.ReLU()
        elif nonlin == "linear":
            nonlin = utils.Linear()
        self.nonlins = [nonlin] * (self.n_layers - 1)
        self.use_bias = use_bias

        # initialize nodes
        self.val_nodes = [[] for _ in range(self.n_layers)]
        self.errs = [[] for _ in range(self.n_layers)]

        # sparse penalty
        self.lamb = lamb

    def set_sparsity(self, sparsity):
        self.lamb = sparsity

    def get_inf_losses(self):
        return self.inf_losses  # inf_iters,

    def update_err_nodes(self):
        for l in range(0, self.n_layers):
            if l == 0:
                self.errs[l] = self.val_nodes[l] - self.memory
            else:
                preds = self.layers[l - 1](self.nonlins[l - 1](self.val_nodes[l - 1]))
                self.errs[l] = self.val_nodes[l] - preds

    def update_val_nodes(self, inf_lr):
        for l in range(0, self.n_layers - 1):
            derivative = self.nonlins[l].deriv(self.val_nodes[l])
            # sparse penalty
            penalty = self.lamb if l == 0 else 0.0
            delta = (
                -self.errs[l]
                - penalty * torch.sign(self.val_nodes[l])
                + derivative * torch.matmul(self.errs[l + 1], self.layers[l].weight)
            )
            self.val_nodes[l] = (
                F.relu(self.val_nodes[l] + inf_lr * delta)
                if self.relu_inf
                else self.val_nodes[l] + inf_lr * delta
            )

    def set_nodes(self, batch_inp):
        # computing val nodes
        self.val_nodes[0] = self.memory.clone()
        for l in range(1, self.n_layers - 1):
            self.val_nodes[l] = self.layers[l - 1](
                self.nonlins[l - 1](self.val_nodes[l - 1])
            )
        self.val_nodes[-1] = batch_inp.clone()

        # computing error nodes
        self.update_err_nodes()

    def inference(self, batch_inp, n_iters, inf_lr):
        self.set_nodes(batch_inp)
        self.batch_size = batch_inp.shape[0]
        self.inf_losses = []

        for itr in range(n_iters):
            with torch.no_grad():
                self.update_val_nodes(inf_lr)
            self.update_err_nodes()
            self.inf_losses.append(self.get_energy().item())

    def get_energy(self):
        """Function to obtain the sum of all layers' squared MSE"""
        total_energy = 0
        for l in range(self.n_layers):
            total_energy += torch.sum(
                self.errs[l] ** 2
            )  # average over batch and feature dimensions
        return total_energy * (100 / self.batch_size)
