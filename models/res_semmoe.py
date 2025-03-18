import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
import torch.nn.functional as F
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=2):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels//reduction, bias=False),
            nn.LeakyReLU(inplace=False),
            nn.Linear(in_channels//reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size, stride=stride,
                      padding=padding, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=False),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=k_size, stride=stride,
                      padding=padding, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=False)
        )

        self.se = SEBlock(out_channels)

        self.shortcut = nn.Conv3d(in_channels=in_channels,  out_channels=out_channels, kernel_size=(1, 1, 1), stride=stride,
                                  bias=False)

        self.concat = nn.Sequential(
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):
        x1 = self.layer(x)
        x1 = self.se(x1)
        x2 = self.shortcut(x)
        x = self.concat(x1 + x2)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, root_feat_maps=16, pool_size=2, p=0.3):
        super(Encoder, self).__init__()
        # from grad_demo import backward_hook
        # self.register_backward_hook(backward_hook)
        self.first = ConvBlock(in_channels, root_feat_maps)
        self.down1 = nn.Sequential(
            nn.MaxPool3d(pool_size),
            ConvBlock(root_feat_maps, root_feat_maps * 2)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool3d(pool_size),
            ConvBlock(root_feat_maps * 2, root_feat_maps * 4)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool3d(pool_size),
            ConvBlock(root_feat_maps * 4, root_feat_maps * 8)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool3d(pool_size),
            ConvBlock(root_feat_maps * 8, root_feat_maps * 16)
        )

        self.down5 = nn.Sequential(
            nn.MaxPool3d(pool_size),
            ConvBlock(root_feat_maps * 16, root_feat_maps * 32)
        )

        self.drop = nn.Dropout3d(p=p, inplace=True)

    def forward(self, x):
        out1 = self.first(x)
        out2 = self.down1(out1)
        out3 = self.down2(out2)
        out4 = self.down3(out3)
        out4 = self.drop(out4)
        out5 = self.down4(out4)
        out5 = self.drop(out5)
        out6 = self.down5(out5)
        out6 = self.drop(out6)

        res = [out1, out2, out3, out4, out5, out6]

        return res

class DecBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mode_upsample=0):
        super().__init__()
        self.mode = mode_upsample
        if mode_upsample == 0:
            self.down = nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2, bias=False),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(inplace=False)
            )
            self.conv = ConvBlock(in_channels, out_channels)
        elif mode_upsample == 1:
            self.down = nn.Sequential(
                nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True),
            )
            self.conv = ConvBlock(in_channels + out_channels, out_channels)
        elif mode_upsample == 2:
            self.down = nn.Sequential(
                nn.Upsample(scale_factor=(2, 2, 2), mode='nearest'),
            )
            self.conv = ConvBlock(in_channels + out_channels, out_channels)

    def forward(self, x, y):
        x = self.down(x)
        x = torch.cat([y, x], dim=1)
        x = self.conv(x)
        return x

class Decoder(nn.Module):
    def __init__(self, root_feat_maps=16, mode_upsample=0):
        super(Decoder, self).__init__()
        # from grad_demo import backward_hook
        # self.register_backward_hook(backward_hook)
        self.up5 = DecBlock(root_feat_maps * 32, root_feat_maps * 16, mode_upsample)
        self.up4 = DecBlock(root_feat_maps * 16, root_feat_maps * 8, mode_upsample)
        self.up1 = DecBlock(root_feat_maps * 8, root_feat_maps * 4, mode_upsample)
        self.up2 = DecBlock(root_feat_maps * 4, root_feat_maps * 2, mode_upsample)
        self.up3 = DecBlock(root_feat_maps * 2, root_feat_maps, mode_upsample)

    def forward(self, res):
        out1 = res[0]
        out2 = res[1]
        out3 = res[2]
        out4 = res[3]
        out5 = res[4]
        out6 = res[5]
        out = self.up5(out6, out5)
        out = self.up4(out, out4)
        out = self.up1(out, out3)
        out = self.up2(out, out2)
        out = self.up3(out, out1)
        return out

class FinalConv(nn.Module):
    def __init__(self, out_channels, root_feat_maps=16):
        super(FinalConv, self).__init__()
        # from grad_demo import backward_hook
        # self.register_backward_hook(backward_hook)
        self.final = nn.Conv3d(root_feat_maps, out_channels, 1)

    def forward(self, x):
        x = self.final(x)
        a = self.final.weight
        return x


class SparseDispatcher(object):

    def __init__(self, num_experts,task_num, gates):
        """Create a SparseDispatcher."""
        # TODO _gates 的作用 ？
        self._gates = gates
        self._num_experts = num_experts
        self._task_num = task_num
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 2], 0]
        self._task_index = torch.nonzero(gates)[index_sorted_experts[:, 2], 1]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        #新添加
        # self._part_sizes_sum = [(self._part_sizes[0][i] + self._part_sizes[1][i]) for i in range(self._num_experts)]
        self._part_sizes_sum = list(np.array(self._part_sizes).sum(axis = 0))
        # expand gates to match with self._batch_index
        self._batch_split = torch.split(self._batch_index, self._part_sizes_sum, dim=0)
        self._batch_split_dewe = [torch.unique(self._batch_split[i]) for i in range(self._num_experts)]
        self._patch_dewe = [len(self._batch_split_dewe[i]) for i in range(self._num_experts)]
        self._batch_dewe = torch.cat(self._batch_split_dewe, 0)
        self._batch_task_list = []
        for i in range(self._task_num):
            self._batch_task_list.append([])
        for index in range(self._batch_index.size()[0]):
            self._batch_task_list[self._task_index[index]].append(int(self._batch_index[index]))
        self._batch_task = []
        for i in range(self._task_num):
            self._batch_task.append(torch.tensor(self._batch_task_list[i]))
        # self._batch_task = [torch.tensor(self._batch_task_list[0]), torch.tensor(self._batch_task_list[1])]
        gates_exp = self._gates.reshape(self._gates.size()[0], self._task_num * self._num_experts)[self._batch_index.flatten()]
        #修改：
        index = torch.tensor([int(self._batch_index[i]) * self._task_num + int(self._task_index[i]) for i in range(self._batch_index.size()[0])])
        gates_exp = gates.reshape(gates.size()[0] * self._task_num, self._num_experts)[index]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

        # expand gates to match with self._batch_index
        # gates_exp = gates[self._batch_index.flatten()]
        # self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

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
        input = dict()
        for i in range(0,self._num_experts):
            input[i] = []
        inp_exp = [inp[i][self._batch_dewe].squeeze(1) for i in range(0,6)]   #6: encoder的输出是6个tensor
        for i in range(0,6):
            temp = torch.split(inp_exp[i], self._patch_dewe, dim=0)
            for j in range(0,self._num_experts):
                input[j].append(temp[j])
        result = []
        for i in range(0,self._num_experts):
            result.append(input[i])
        return tuple(result)

    def combine(self, expert_out, multiply_by_gates=True):
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
        # stitched = torch.cat(expert_out, 0).exp()   就是把结果拼接在一起 为后续分配给每个batch做准备
        stitched = None
        for exp in range(self._num_experts):
            a = dict()
            for index in range(self._batch_split_dewe[exp].size()[0]):
                a[int(self._batch_split_dewe[exp][index])] = index
            for index in self._batch_split[exp]:
                if stitched == None:
                    stitched = expert_out[exp][a[int(index)]].unsqueeze(0)
                else:
                    stitched = torch.cat((stitched, expert_out[exp][a[int(index)]].unsqueeze(0)), 0)
        # 门控机制与下面代码相关
        size = [stitched.size()[0],stitched.size()[1],stitched.size()[2],stitched.size()[3],stitched.size()[4]]
        if multiply_by_gates:
            stitched = stitched.reshape(stitched.size()[0],stitched.size()[1]*stitched.size()[2]*stitched.size()[3]*stitched.size()[4])
            result = stitched.mul(self._nonzero_gates).reshape(size[0],size[1],size[2],size[3],size[4])
        #修改后
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), expert_out[-1].size(2), expert_out[-1].size(3), expert_out[-1].size(4), requires_grad=True, device=stitched.device)
        task_list = []
        for index in range(self._batch_index.size()[0]):
            task_list.append(int(self._task_index[index]))
        task_list = torch.tensor(task_list)
        index = dict()
        data = dict()
        for i in range(self._task_num):#对于多个任务来说  获取对应任务的索引 然后将数据分给每个任务
            index[i] = torch.nonzero(task_list == i).squeeze()
            data[i] = torch.index_select(result, 0, index[i].to(device))
        combined = torch.cat((zeros.index_add(0, self._batch_task[0].cuda(), data[0].float()).unsqueeze(1),zeros.index_add(0, self._batch_task[1].cuda(), data[1].float()).unsqueeze(1)), 1)
        for i in range(2,self._task_num):
            combined = torch.cat((combined,zeros.index_add(0, self._batch_task[i].cuda(), data[i].float()).unsqueeze(1)), 1)
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined
        # return combined.log()

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MMoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, input_size, in_channels, out_channels, num_experts, noisy_gating=True, k=2, task_num=2, attr= [0,0,1]):
        super(MMoE, self).__init__()
        self.root_feat_maps = 12
        self.noisy_gating = noisy_gating
        self.attr = attr
        self.num_experts = num_experts
        self.in_channels = in_channels
        self.input_size = input_size
        self.out_channels = out_channels
        self.k = k
        self.task_num = task_num
        # instantiate experts
        self.encoder = Encoder(self.in_channels, self.root_feat_maps, pool_size=2, p=0.3)
        # self.tower = nn.ModuleList([FinalConv(self.out_channels, self.root_feat_maps) for i in range(self.task_num)])
        self.finalorganconv = FinalConv(4, self.root_feat_maps)
        self.finaltumorconv = FinalConv(2, self.root_feat_maps)
        self.experts = nn.ModuleList([Decoder(self.root_feat_maps, mode_upsample=0) for i in range(self.num_experts)])

        # self.w_gate = nn.Parameter(torch.zeros(512*3*3*3, task_num, num_experts), requires_grad=True)
        self.w_gate = nn.Parameter(torch.zeros(49152, task_num, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(49152, task_num, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(2)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        self.attribute = torch.nn.Sequential(torch.nn.Linear(self.task_num, self.task_num*self.num_experts), torch.nn.GELU(),
                            torch.nn.LayerNorm(self.task_num*self.num_experts))
        # from grad_demo import backward_hook
        # self.register_backward_hook(backward_hook)
        assert(self.k <= self.num_experts)

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
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

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
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, 1, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        #输入直接作为门控的输入
        clean_logits = x @ self.w_gate.reshape(x.size()[1],self.task_num*self.num_experts)
        f = sum(self.w_gate == 0)
        clean_logits = clean_logits.reshape(x.size()[0], self.task_num, self.num_experts)
        clean_logits = F.normalize(clean_logits, dim=2)
        #属性向量作为门控输入
        # routing = self.attribute(torch.FloatTensor(self.attr).to(device)).reshape(x.size()[0], self.task_num*self.num_experts)
        # clean_logits = routing @ self.w_gate.reshape(self.task_num * self.num_experts, self.task_num * self.num_experts)
        # clean_logits = clean_logits.reshape(x.size()[0], self.task_num, self.num_experts)
        # clean_logits = F.normalize(clean_logits, dim=2)
        # clean_logits = clean_logits + routing
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise.reshape(x.size()[1],self.task_num*self.num_experts)
            raw_noise_stddev = raw_noise_stddev.reshape(x.size()[0], self.task_num, self.num_experts)
            raw_noise_stddev = F.normalize(raw_noise_stddev, dim=2)
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=2)
        top_k_logits = top_logits[:,:, :self.k]
        top_k_indices = top_indices[:,:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(2, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1e-2):
        """Args:
        x: tensor shape [batch_size, 1, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size,1, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        res = self.encoder(x)
        size = res[5].size()
        gates, load = self.noisy_top_k_gating(res[5].reshape(size[0], size[1]*size[2]*size[3]*size[4]), self.training)
        # print(gates)
        # calculate importance loss
        # importance = gates.sum(0)
        # #
        # loss = self.cv_squared(importance) + self.cv_squared(load)
        # loss *= loss_coef
        loss = 0
        # TODO gate ?
        dispatcher = SparseDispatcher(self.num_experts,self.task_num, gates)
        expert_inputs = dispatcher.dispatch(res)
        # gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        # outputs = [self.tower[i](y[:, i, :, :, :]) for i in range(self.task_num)]
        outputs = [self.finalorganconv(y[:, 0, :, :, :]), self.finaltumorconv(y[:, 1, :, :, :])]
        # print(gates)
        return outputs