
import torch
import torch.nn.functional as F
import math
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
import torch
import math

class adapter(nn.Module):
    def __init__(self, embed_dim=512, hidden_dim=32):
        super().__init__()

        self.adapter_down = nn.Linear(embed_dim*2, hidden_dim)
        self.adapter_up = nn.Linear(hidden_dim, embed_dim*2)
        self.adapter_mid = nn.Linear(hidden_dim, hidden_dim)

        nn.init.kaiming_uniform_(self.adapter_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.kaiming_uniform_(self.adapter_mid.weight, a=math.sqrt(5))
        nn.init.zeros_(self.adapter_mid.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = embed_dim

    def forward(self, x):
        B, N, C = x.shape
        x_down = self.adapter_down(x)
        x_down = self.act(x_down)
        x_down = self.adapter_mid(x_down)
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)
        return x_up







class adapter_down(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=32):
        super().__init__()

        self.adapter_down = nn.Linear(embed_dim, hidden_dim)

        self.adapter_mid = nn.Linear(hidden_dim, hidden_dim)

        #nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_mid.bias)
        nn.init.zeros_(self.adapter_mid.weight)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)


        #self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = embed_dim

    def forward(self, x):
        B, N, C = x.shape
        x_down = self.adapter_down(x)
        #x_down = self.act(x_down)
        x_down = self.adapter_mid(x_down)
        #x_down = self.act(x_down)
        x_down = self.dropout(x_down)

        #print("return adap x", x_up.size())
        return x_down



class adapter_up(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=32):
        super().__init__()


        self.adapter_up = nn.Linear(hidden_dim, embed_dim)

        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)


        self.dropout = nn.Dropout(0.1)
        self.dim = embed_dim

    def forward(self, x):
        x_up = self.adapter_up(x)
        return x_up


class adapter_copy(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        _, _, C = x.shape
        tensor_1 = x[:,:,:int(C/2)]
        tensor_2 = x[:,:,int(C/2):]


        return tensor_1+tensor_2



class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, top_logits, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates.unsqueeze(1))
            # stitched = stitched.mul(top_logits.unsqueeze(1))
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), expert_out[-1].size(2), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)





class MoEFusion(nn.Module):
    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, input_size=768, patch_num_x = 256,num_experts=8, noisy_gating=True, k=2, patch_num_z=64, moe_type=None, moe_cfg=None):
        super(MoEFusion, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts

        self.input_size = input_size
        self.k = k
        self.moe_type = moe_type
        self.patch_num_x = patch_num_x
        self.patch_num_z = patch_num_z
        # BMR-HMoE mechanism ablation switches (safe defaults = full method).
        self.substitute_mode = getattr(moe_cfg, "SUBSTITUTE_MODE", "hallucinate")
        self.use_recon_loss = getattr(moe_cfg, "USE_RECON_LOSS", True)
        self.hallucinate_direction = getattr(moe_cfg, "HALLUCINATE_DIRECTION", "bilateral")
        self.use_ortho = getattr(moe_cfg, "USE_ORTHO", True)

        # rank = 64
        # emb = 768
        # instantiate experts
        # self.experts = nn.ModuleList([adapter(768*2,4*int(i)) for i in range(int(self.num_experts))]
        #                 )
        # self.experts = nn.ModuleList([adapter(512,4),adapter(512,8),adapter(512,16),adapter(512,32), adapter(512,64),adapter(512,128),adapter(512,256),adapter(512,512)])

        if moe_type == "BMR_HMoE":
            # PAMI Extension: Bilateral Modality-Specific Feature Reconstruction & Hallucination with Heterogeneous MoE Gating
            # 1. 8 Heterogeneous experts with exponential capacity (rank 2 is removed, starting at 4):
            self.num_experts = 8
            self.experts = nn.ModuleList([
                adapter(self.input_size, 4), adapter(self.input_size, 8), adapter(self.input_size, 16), adapter(self.input_size, 32),
                adapter(self.input_size, 64), adapter(self.input_size, 128), adapter(self.input_size, 256), adapter(self.input_size, self.input_size)
            ])
            # 2. Modality Hallucination / Reconstruction Networks (MSRH)
            # Reconstructs/hallucinates the representation of the missing modality from the available modality
            self.rgb_to_aux_hallucinater = nn.Sequential(
                nn.Linear(self.input_size, 128),
                nn.GELU(),
                nn.Linear(128, self.input_size)
            )
            self.aux_to_rgb_hallucinater = nn.Sequential(
                nn.Linear(self.input_size, 128),
                nn.GELU(),
                nn.Linear(128, self.input_size)
            )
            # Initialize hallucinater to zero so at training start it behaves stably
            nn.init.zeros_(self.rgb_to_aux_hallucinater[-1].weight)
            nn.init.zeros_(self.rgb_to_aux_hallucinater[-1].bias)
            nn.init.zeros_(self.aux_to_rgb_hallucinater[-1].weight)
            nn.init.zeros_(self.aux_to_rgb_hallucinater[-1].bias)
            
            # Gating parameters
            total_template_tokens = self.patch_num_z * 5
            self.w_down_template = nn.Parameter(torch.zeros(total_template_tokens, self.patch_num_z), requires_grad=True)
            self.w_gate = nn.Parameter(torch.zeros(self.patch_num_x + self.patch_num_z, self.num_experts), requires_grad=True)
            self.w_noise = nn.Parameter(torch.zeros(self.patch_num_x + self.patch_num_z, self.num_experts), requires_grad=True)
            torch.nn.init.xavier_normal_(self.w_gate)
            
            self.softplus = nn.Softplus()
            self.softmax = nn.Softmax(1)
            self.register_buffer("mean", torch.tensor([0.0]))
            self.register_buffer("std", torch.tensor([1.0]))
            assert (self.k <= self.num_experts)
        elif moe_type == "BIG":
            self.experts = nn.ModuleList([adapter(512,512),adapter(512,512),adapter(512,512),adapter(512,512), adapter(512,512),adapter(512,512),adapter(512,512),adapter(512,512)])
        elif moe_type == "SMALL":
            self.experts = nn.ModuleList([adapter(512,4),adapter(512,4),adapter(512,4),adapter(512,4), adapter(512,4),adapter(512,4),adapter(512,4),adapter(512,4)])
        elif moe_type == "MIDDLE":   
            self.experts = nn.ModuleList([adapter(512,128),adapter(512,128),adapter(512,128),adapter(512,128), adapter(512,128),adapter(512,128),adapter(512,128),adapter(512,128)])
        elif moe_type == "4E":   
            self.num_experts = 4
            self.experts = nn.ModuleList([adapter(512,4),adapter(512,8),adapter(512,16),adapter(512,32)])
        elif moe_type == "NEW_DIFFERENT":   
            self.num_experts = 8
            self.experts = nn.ModuleList([adapter(512,2),adapter(512,4),adapter(512,8),adapter(512,16),adapter(512,32), adapter(512,64),adapter(512,128),adapter(512,256),adapter(512,512)])
        elif moe_type == "HYBRID_NEW":
            self.num_experts = 8
            self.experts = nn.ModuleList([adapter(512,4),adapter(512,4),adapter(512,6),adapter(512,6), adapter(512,8),adapter(512,8),adapter(512,10),adapter(512,10)])
        elif moe_type == "HYBRID":
            self.num_experts = 8
            self.experts = nn.ModuleList([adapter(512,4),adapter(512,4),adapter(512,8),adapter(512,8), adapter(512,12),adapter(512,12),adapter(512,16),adapter(512,16)])
        elif moe_type == "NONE":
            # Tier-2/3 ablation: no MoE, no BMR hallucination/gating. Pure residual
            # fusion over the concatenated [rgb|aux] features (dim 2*input_size).
            # Zero-initialised so it starts as identity (matching the adapter experts'
            # zero-init up-projection), then learns a residual. Missing modalities are
            # left zeroed (plain masking, no hallucination/reconstruction loss). The
            # expert/gating params created below are unused here and harmless under
            # DDP find_unused_parameters=True.
            self.none_fusion = nn.Linear(self.input_size * 2, self.input_size * 2)
            nn.init.zeros_(self.none_fusion.weight)
            nn.init.zeros_(self.none_fusion.bias)
            self.experts = nn.ModuleList([adapter(512,4),adapter(512,6),adapter(512,8),adapter(512,10), adapter(512,12),adapter(512,14),adapter(512,16),adapter(512,18)])
        else:
            self.experts = nn.ModuleList([adapter(512,4),adapter(512,6),adapter(512,8),adapter(512,10), adapter(512,12),adapter(512,14),adapter(512,16),adapter(512,18)])


        self.shared_experts = adapter(self.input_size,2)


        # self.experts_1 = nn.ModuleList([adapter(768*2,16) for i in range(int(self.num_experts/2))])

        self.patch_num_z = patch_num_z



        self.balance = nn.Parameter(torch.ones(8),requires_grad=True)



        # self.shared_expert = adapter()



        # self.w_gate = nn.Parameter(torch.zeros(441, self.num_experts), requires_grad=True)
        # self.w_noise = nn.Parameter(torch.zeros(441, self.num_experts), requires_grad=True)
        total_template_tokens = self.patch_num_z * 5
        self.w_down_template = nn.Parameter(torch.zeros(total_template_tokens, self.patch_num_z), requires_grad=True)
        self.w_gate = nn.Parameter(torch.zeros(self.patch_num_x + self.patch_num_z, self.num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(self.patch_num_x + self.patch_num_z, self.num_experts), requires_grad=True)

        torch.nn.init.xavier_normal_(self.w_gate)


        # self.w_gate_1 = nn.Parameter(torch.zeros(256, num_experts), requires_grad=True)
        # self.w_noise_1 = nn.Parameter(torch.zeros(256, num_experts), requires_grad=True)


        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert (self.k <= self.num_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        # Load = number of examples routed to each expert (gate > 0). (A prior
        # positive per-expert scaling before this test was a no-op and was removed.)
        return gates.gt(0).sum(dim=0)

        return gates

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        gate_1,_ = torch.max(x, dim=-1)
        gate_1 = gate_1.squeeze(-1)
        # gate_2 = self.router(z)
        gate_1_template = gate_1[:,self.patch_num_x:]@self.w_down_template

        gate_1 = torch.concat((gate_1[:,:self.patch_num_x],gate_1_template),dim=-1)

        clean_logits = gate_1.squeeze(-1) @ self.w_gate

        # clean_logits = clean_logits+ self.rank_bias
        
        if self.noisy_gating and train:
            raw_noise_stddev = gate_1.squeeze(-1)  @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)


        top_logits_clean, top_indices_clean = clean_logits.topk(min(self.k + 1, self.num_experts), dim=1)

        # print(top_indices_clean[:,:-1].tolist())

        zeros = torch.zeros_like(logits, requires_grad=True).to(top_k_gates.dtype)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        if train:
            return gates, load, self.softmax(logits), top_k_gates, torch.mean(self.softmax(noisy_logits)/torch.norm(self.softmax(noisy_logits), dim=-1, keepdim=True),dim=0)
        else:
            return gates, load, self.softmax(logits), top_k_gates, torch.mean(self.softmax(clean_logits)/torch.norm(self.softmax(clean_logits), dim=-1, keepdim=True),dim=0)

    

    def forward(self, x,  loss_coef=1e-2, missing_mask=None):
        if self.moe_type == "NONE":
            # Pure residual fusion, no experts/gating/hallucination. Aux loss = 0.
            return x + self.none_fusion(x), x.new_zeros((), dtype=x.dtype)
        if self.moe_type == "BMR_HMoE":
            b, l, c = x.shape
            x_rgb = x[:, :, :self.input_size]
            x_aux = x[:, :, self.input_size:]
            
            # 1. Bilateral Modality Hallucination & Feature Reconstruction (MSRH)
            # Reconstruct missing RGB features from Aux, and Aux from RGB.
            # HALLUCINATE_DIRECTION ablation: disable one direction by zeroing its
            # hallucinated output (so that path contributes neither substitution
            # nor reconstruction loss).
            hallucinated_rgb = self.aux_to_rgb_hallucinater(x_aux)   # aux -> rgb
            hallucinated_aux = self.rgb_to_aux_hallucinater(x_rgb)   # rgb -> aux
            if self.hallucinate_direction == "rgb2aux":
                hallucinated_rgb = torch.zeros_like(hallucinated_rgb)
            elif self.hallucinate_direction == "aux2rgb":
                hallucinated_aux = torch.zeros_like(hallucinated_aux)

            # Detect missing modalities dynamically using passed mask if available
            if missing_mask is not None:
                rgb_is_missing = (missing_mask[:, 0:1] == 0.0).unsqueeze(-1) # [B, 1, 1]
                aux_is_missing = (missing_mask[:, 1:2] == 0.0).unsqueeze(-1) # [B, 1, 1]
            else:
                rgb_is_missing = (x_rgb.abs().mean(dim=(1, 2)) < 1e-4).unsqueeze(-1).unsqueeze(-1) # [B, 1, 1]
                aux_is_missing = (x_aux.abs().mean(dim=(1, 2)) < 1e-4).unsqueeze(-1).unsqueeze(-1) # [B, 1, 1]

            # Substitute missing modality features. SUBSTITUTE_MODE ablation:
            # "hallucinate" (default) fills with cross-modal features; "zero"
            # leaves them zeroed (plain masking baseline).
            if self.substitute_mode == "zero":
                fill_rgb = torch.zeros_like(x_rgb)
                fill_aux = torch.zeros_like(x_aux)
            else:
                fill_rgb = hallucinated_rgb
                fill_aux = hallucinated_aux
            r_rgb = torch.where(rgb_is_missing, fill_rgb, x_rgb)
            r_aux = torch.where(aux_is_missing, fill_aux, x_aux)

            # Bilateral consistency loss (only when the target modality is present/healthy).
            # USE_RECON_LOSS ablation gates the whole term.
            recon_loss = 0.0
            if self.training and self.use_recon_loss:
                rgb_present_mask = (~rgb_is_missing).float()
                aux_present_mask = (~aux_is_missing).float()

                loss_rgb = (F.mse_loss(hallucinated_rgb, x_rgb, reduction='none') * rgb_present_mask).mean()
                loss_aux = (F.mse_loss(hallucinated_aux, x_aux, reduction='none') * aux_present_mask).mean()
                recon_loss = (loss_rgb + loss_aux) * loss_coef
            
            # 2. Combined representation for gating and experts
            x_reconstructed = torch.cat((r_rgb, r_aux), dim=-1) # [B, L, 1024]
            
            # 3. Gating routing to 8 Heterogeneous experts
            gates, load, logits, top_k_logits, logits_mean = self.noisy_top_k_gating(x_reconstructed, self.training)
            
            importance = gates.sum(0)
            loss_balance_new = torch.sum(((logits_mean - 1 / self.k) ** 2))
            moe_loss = self.cv_squared(load) + loss_balance_new
            moe_loss *= loss_coef
            
            # Orthogonality constraint on gating router to ensure non-redundant
            # complementary paths. USE_ORTHO ablation drops this term.
            if self.use_ortho:
                ortho_reg = torch.norm(torch.matmul(self.w_gate, self.w_gate.T) - torch.eye(self.patch_num_x + self.patch_num_z, device=x.device)) * 1e-4
            else:
                ortho_reg = 0.0

            total_moe_loss = moe_loss + recon_loss + ortho_reg
            
            dispatcher = SparseDispatcher(self.num_experts, gates)
            expert_inputs_x = dispatcher.dispatch(x_reconstructed)
            
            # Route tokens to the 8 heterogeneous experts (widths: 4 to 512)
            expert_outputs_x = [self.experts[i](expert_inputs_x[i]) for i in range(self.num_experts)]
            y_x = dispatcher.combine(expert_outputs_x, top_k_logits)
            
            return y_x, total_moe_loss
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses
        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        b,_,_ = x.shape

        gates, load, logits,top_k_logits,logits_mean = self.noisy_top_k_gating(x, self.training)

        if self.moe_type == "HYBRID":
            penalty = [0.4,0.4,0.8,0.8,1.2,1.2,1.6,1.6]
            penalty=torch.tensor(penalty)
            penalty = penalty.unsqueeze(0).expand(gates.shape[0], -1).to(gates.device)
            importance = (gates*penalty).sum(0)
            loss_balance_new = torch.sum(((logits_mean-1/self.k)**2))
        elif  self.moe_type == "DIFFERENT":
            penalty = [4,6,8,10,12,14,16,18]
            penalty=torch.tensor(penalty)
            penalty = penalty.unsqueeze(0).expand(gates.shape[0], -1).to(gates.device)
            importance = (gates*penalty).sum(0)
            loss_balance_new = torch.sum(((logits_mean-1/self.k)**2))
        else:
            importance = (gates*self.balance).sum(0)
            loss_balance_new = torch.sum(((logits_mean-1/self.k)**2))
        importance = (gates).sum(0)
        #
        # print(gates)

        # loss = self.cv_squared(importance) + self.cv_squared(load)
        loss = self.cv_squared(load) + loss_balance_new 

        loss *= loss_coef



        # if gates.argmax()>=4:
        #     print("yes")

        # 将 Tensor 移动到 CPU 上（便于后续操作）
        # tensor = gates.cpu()

        # # 使用 nonzero 找到非零元素的索引
        # indices = torch.nonzero(tensor)

        # # 将索引转换为 Python 列表remotec
        # indices_list = indices.tolist()
        # print(indices_list)




        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs_x = dispatcher.dispatch(x)
        expert_outputs_x = [self.experts[i](expert_inputs_x[i]) for i in range(self.num_experts)]
        y_x = dispatcher.combine(expert_outputs_x, top_k_logits)



        # gates_1, load_1, logits_1,top_k_logits_1 = self.noisy_top_k_gating(x[:,64:,:], self.training)
        # importance_1 = gates_1.sum(0)
        # loss_1 = self.cv_squared(importance_1)+ self.cv_squared(load_1)
        # loss_1 *= loss_coef
        # dispatcher_1 = SparseDispatcher(self.num_experts, gates_1)
        # expert_inputs_x_1 = dispatcher_1.dispatch(y_x)
        # expert_outputs_x_1 = [self.experts_1[i](expert_inputs_x_1[i]) for i in range(self.num_experts)]
        # y_x = dispatcher.combine(expert_outputs_x_1, top_k_logits)



        return y_x, loss





if "__main__"==__name__:
    moe_instance = MoEFusion(input_size=256, num_experts=4)
    tensor_1 = torch.randn(64,64,768)
    tensor_2 = torch.randn(64,256,768)

    xx = moe_instance(tensor_1,tensor_2)
