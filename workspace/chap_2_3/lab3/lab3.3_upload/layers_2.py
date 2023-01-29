# coding:utf-8
import numpy as np
import struct
import os
import time

"""
加速思路：

原算法使用（四重循环中的最后）二重循环来描述conv_window 和 pool_window在input上的滑动。循环意味着计算是定序的，串行的。
尽管循环中两次计算间的数据是不相关的，但编译器没有智能到发现这种数据的无关性并进行并行优化，而是只会按照给定的循环顺序进行单步计算。

优化方法为：将循环中每次计算用到的数据单独提取出来，依次放在一个大矩阵中，然后用该矩阵参与运算。
矩阵中的数据显然是不相关的，所以编译器就可以“放心大胆地”对该运算进行并行优化，大大提高了运算速度。
"""

def img2col(input, h_out, w_out, kernel_size, stride):
    """ 
        Transfer the feature maps to columns.
        Args:
            input:      The input feature map, [N, C, H, W].
            h_out:      The output height.
            w_out:      The output width.
            kernel_size:The kernel size.
            stride:     The stride.
        Return:
            output:     [N, C, kernel_size*kernel_size, h_out*w_out]        
    """
    N, C, H, W = input.shape
    output = np.zeros([N, C, kernel_size * kernel_size, h_out * w_out])
    _height = (H - kernel_size) // stride + 1
    _width = (W - kernel_size) // stride + 1
    
    #for idxn in range(N):
        # For each channel, scan from left-top to right-bottom
    for idxh in range(_height):
        for idxw in range(_width):
            output[:, :, :, idxh * _width+idxw] = \
                input[:, :, idxh * stride:idxh*stride + kernel_size, idxw * stride : idxw * stride + kernel_size].reshape(N, C, -1)

    return output

def col2img(col_input, h_pad, w_pad, kernel_size, channel, padding, stride):
    """  
        Transfer the columns to feature maps.
        Args:
            col_input:      The input columns, [N, C * kernel_size * kernel_size, h_out * w_out].
            h_pad:          H + 2 * padding.
            w_pad:          W + 2 * padding.
            kernel_size:    The kernel size.
            channel:        Channel in.
            padding:        The padding.
            stride:         The stride.
        Return:
            output:         [N, C, H, W]
    """
    N = col_input.shape[0]
    output_pad = np.zeros([N, channel, h_pad, w_pad])
    # reshape the col_input. [N, C*kernel_size*kernel_size, h_out*w_out] -> [N, C, kernel_size*kernel_size, h_out*w_out]
    _col_input = col_input.reshape(N, channel, -1, col_input.shape[2])
    _height = (h_pad - kernel_size) // stride + 1
    _width = (w_pad - kernel_size) // stride + 1
    for idxh in range(_height):
        for idxw in range(_width):
            output_pad[:, :, idxh * stride : idxh * stride + kernel_size, idxw * stride : idxw * stride + kernel_size] += \
                _col_input[:, :, :, idxh * _width + idxw].reshape(N, channel, kernel_size, -1)
    output = output_pad[:, :, padding : h_pad-padding, padding : w_pad-padding]
    return output 

class ConvolutionalLayer(object):
    def __init__(self, kernel_size, channel_in, channel_out, padding, stride, type=1):
        self.kernel_size = kernel_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.padding = padding
        self.stride = stride
        self.forward = self.forward_raw
        self.backward = self.backward_raw
        
        if type == 1:  # type 设为 1 时，使用优化后的 foward 和 backward 函数
            self.forward = self.forward_speedup
            self.backward = self.backward_speedup
        
        print('\tConvolutional layer with kernel size %d, input channel %d, output channel %d.' % (self.kernel_size, self.channel_in, self.channel_out))
    def init_param(self, std=0.01):
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.channel_in, self.kernel_size, self.kernel_size, self.channel_out))
        self.bias = np.zeros([self.channel_out])

    def forward_raw(self, input):
        start_time = time.time()
        self.input = input # [N, C, H, W]
        height = self.input.shape[2] + self.padding * 2
        width = self.input.shape[3] + self.padding * 2
        self.input_pad = np.zeros([self.input.shape[0], self.input.shape[1], height, width])
        self.input_pad[:, :, self.padding:self.padding+self.input.shape[2], self.padding:self.padding+self.input.shape[3]] = self.input
        height_out = (height - self.kernel_size) / self.stride + 1
        width_out = (width - self.kernel_size) / self.stride + 1
        self.output = np.zeros([self.input.shape[0], self.channel_out, height_out, width_out])
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.channel_out):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        # TODO: 计算卷积层的前向传播，特征图与卷积核的内积再加偏置
                        self.output[idxn, idxc, idxh, idxw] = np.sum(
                            self.weight[:, :, :, idxc] * self.input_pad[idxn, :, idxh * self.stride : idxh * self.stride + self.kernel_size, 
                            idxw * self.stride : idxw * self.stride + self.kernel_size]) + self.bias[idxc]

        self.forward_time = time.time() - start_time
        return self.output

    def forward_speedup(self, input):
    # TODO: 改进forward函数，使得计算加速
        start_time = time.time()
        self.input = input # [N, C, H, W]
        height = self.input.shape[2] + self.padding * 2
        width = self.input.shape[3] + self.padding * 2
        self.input_pad = np.zeros([self.input.shape[0], self.input.shape[1], height, width])
        self.input_pad[:, :, self.padding : self.padding + self.input.shape[2], self.padding : self.padding + self.input.shape[3]] = self.input
        height_out = (height - self.kernel_size) / self.stride + 1
        width_out = (width - self.kernel_size) / self.stride + 1

        # self.col_weights = self.weight.transpose(3, 0, 1, 2).reshape(self.weight.shape[-1], -1)
        """
        old version

        self.weight_reshape = np.reshape(self.weight, [-1, self.channel_out])
        
        self.img2col = np.zeros([self.input.shape[0] * height_out * width_out, self.channel_in * self.kernel_size * self.kernel_size])

        for idxn in range(self.input.shape[0]):
            for idxh in range(height_out):
                for idxw in range(width_out):
                    self.img2col[idxn * height_out * width_out + idxh * width_out + idxw, :] = \
                        self.input_pad[idxn, :, idxh * self.stride : idxh * self.stride + self.kernel_size, 
                            idxw * self.stride : idxw * self.stride + self.kernel_size].reshape([-1])
        
        output = np.matmul(self.img2col, self.weight_reshape) + self.bias
        self.output = output.reshape([self.input.shape[0], height_out, width_out, self.channel_out]).transpose([0, 3, 1, 2])

        self.forward_time = time.time() - start_time
        return self.output
        """
        # new version

        # img2col. [N, C, kernel_size * kernel_size, h_out * w_out]
        self.col_input = img2col(self.input_pad, height_out, width_out, self.kernel_size, self.stride)
        # merge channel. [N, C * kernel_size * kernel_size, h_out * w_out]
        self.col_input = self.col_input.reshape(self.col_input.shape[0], -1, self.col_input.shape[3])
        # reshape kernel. [C_in, kernel_size, kernel_size, C_out] -> [C_out, C_in, kernel_size, kernel_size] -> [C_out, C_in * kernel_size * kernel_size]
        self.col_weights = self.weight.transpose(3, 0, 1, 2).reshape(self.weight.shape[-1], -1)
        # compute convolution.
        output = np.matmul(self.col_weights, self.col_input) + self.bias.reshape(-1, 1)
        self.output = output.reshape(self.input.shape[0], self.channel_out, height_out, width_out)

        self.forward_time = time.time() - start_time
        return self.output
    
    def backward_speedup(self, top_diff):
    # TODO: 改进backward函数，使得计算加速
        start_time = time.time()
        height = self.input.shape[2] + self.padding * 2
        width = self.input.shape[3] + self.padding * 2
        bottom_diff = np.zeros(self.input_pad.shape)
    
        _top_diff = top_diff.transpose(1, 2, 3, 0).reshape(self.channel_out, -1)

        """ d_bias """
        """ self.d_bias = np.sum(top_diff, axis=(0, 2, 3)).flatten() """

        """ d_weight """
        """ # [N, C * kernel_size * kernel_size, h_out * w_out] -> [C * kernel_size * kernel_size, N * h_out * w_out]
        X_hat = self.col_input.transpose(1, 2, 0).reshape(self.channel_in*self.kernel_size*self.kernel_size, -1)
        # [N, C_out, h_out, w_out] -> [C_out, N * h_out * w_out]
        _top_diff = top_diff.transpose(1, 2, 3, 0).reshape(self.channel_out, -1)
        self.d_weight = np.matmul(X_hat, _top_diff.T).reshape(self.weight.shape)
        #self.d_weight = np.matmul(_top_diff, X_hat.T).reshape(self.weight.shape) """

        """ bottom_diff """
        # [C_in * kernel_size * kernel_size, N * h_out * w_out]
        col_bottom_diff = np.matmul(self.col_weights.T, _top_diff)
        # [N, C_in * kernel_size * kernel_size, h_out * w_out]
        col_bottom_diff = col_bottom_diff.reshape(col_bottom_diff.shape[0], -1, self.input.shape[0]).transpose(2, 0, 1)
        # col2img. [[N, C, H, W]]
        bottom_diff = col2img(col_bottom_diff, height, width, self.kernel_size, self.channel_in, self.padding, self.stride)

        self.backward_time = time.time() - start_time
        return bottom_diff

    def backward_raw(self, top_diff):
        start_time = time.time()
        self.d_weight = np.zeros(self.weight.shape)
        self.d_bias = np.zeros(self.bias.shape)
        bottom_diff = np.zeros(self.input_pad.shape)
        for idxn in range(top_diff.shape[0]):
            for idxc in range(top_diff.shape[1]):
                for idxh in range(top_diff.shape[2]):
                    for idxw in range(top_diff.shape[3]):
                        # TODO： 计算卷积层的反向传播， 权重、偏置的梯度和本层损失
                        self.d_weight[:, :, :, idxc] += top_diff[idxn, idxc, idxh, idxw] * self.input_pad[idxn, :, 
                                                            idxh * self.stride : idxh * self.stride + self.kernel_size, 
                                                            idxw * self.stride : idxw * self.stride + self.kernel_size]

                        self.d_bias[idxc] += top_diff[idxn, idxc, idxh, idxw]

                        bottom_diff[idxn, :, idxh * self.stride : idxh * self.stride + self.kernel_size, idxw * self.stride : idxw * self.stride + self.kernel_size] += \
                                                            top_diff[idxn, idxc, idxh, idxw] * self.weight[:, :, :, idxc]

        bottom_diff = bottom_diff[:, :, self.padding : self.input.shape[2] + self.padding, self.padding : self.input.shape[3] + self.padding]   

        self.backward_time = time.time() - start_time
        return bottom_diff

    def get_gradient(self):
        return self.d_weight, self.d_bias
    def update_param(self, lr):
        self.weight += - lr * self.d_weight
        self.bias += - lr * self.d_bias
    def load_param(self, weight, bias):
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias
    def get_forward_time(self):
        return self.forward_time
    def get_backward_time(self):
        return self.backward_time

class MaxPoolingLayer(object):
    def __init__(self, kernel_size, stride, type=1):
        self.kernel_size = kernel_size
        self.stride = stride
        self.forward = self.forward_raw
        self.backward = self.backward_raw_book
        if type == 1:  # type 设为 1 时，使用优化后的 foward 和 backward 函数
            self.forward = self.forward_speedup
            self.backward = self.backward_speedup
        print('\tMax pooling layer with kernel size %d, stride %d.' % (self.kernel_size, self.stride))
    def forward_raw(self, input):
        start_time = time.time()
        self.input = input # [N, C, H, W]
        self.max_index = np.zeros(self.input.shape)
        height_out = (self.input.shape[2] - self.kernel_size) / self.stride + 1
        width_out = (self.input.shape[3] - self.kernel_size) / self.stride + 1
        self.output = np.zeros([self.input.shape[0], self.input.shape[1], height_out, width_out])
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.input.shape[1]):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        # TODO： 计算最大池化层的前向传播， 取池化窗口内的最大值
                        self.output[idxn, idxc, idxh, idxw] = np.max(
                            self.input[idxn, idxc, idxh * self.stride : idxh * self.stride + self.kernel_size, 
                            idxw * self.stride : idxw * self.stride + self.kernel_size])

                        curren_max_index = np.argmax(self.input[idxn, idxc, idxh*self.stride:idxh*self.stride+self.kernel_size, 
                                                    idxw*self.stride:idxw*self.stride+self.kernel_size])

                        curren_max_index = np.unravel_index(curren_max_index, [self.kernel_size, self.kernel_size])
                        
                        self.max_index[idxn, idxc, idxh*self.stride+curren_max_index[0], idxw*self.stride+curren_max_index[1]] = 1

        self.forward_time = time.time() - start_time
        return self.output

    def forward_speedup(self, input):
        # TODO: 改进forward函数，使得计算加速
        start_time = time.time()
        self.input = input # [N, C, H, W]
        self.N, self.C, self.H, self.W = self.input.shape
        self.height_out = (self.H - self.kernel_size) // self.stride + 1
        self.width_out = (self.W - self.kernel_size) // self.stride + 1

        # img2col
        # [N, C, H, W] -> [N, C, kernel_size * kernel_size, h_out, w_out]
        self.col_input = img2col(self.input, self.height_out, self.width_out, self.kernel_size, self.stride).reshape(self.N, self.C, -1, self.height_out, self.width_out)
        # [N, C, 1, h_out, w_out]
        output = self.col_input.max(axis=2, keepdims=True) # axis2: each conv_window
        # The index of the max number. [N, C, kernel_size * kernel_size, h_out * w_out]   used in backward
        self.max_index = (self.col_input == output) # the same shape, with 0 or 1
        # [N, C, h_out, w_out]
        self.output = output.reshape(self.N, self.C, self.height_out, self.width_out)

        self.forward_time = time.time() - start_time
        return self.output

    def backward_speedup(self, top_diff):
        # TODO: 改进backward函数，使得计算加速
        start_time = time.time()
        # [N, C * kernel_size * kernel_size, h_out * w_out]
        pool_diff = (self.max_index * top_diff[:, :, np.newaxis, :, :]).reshape(self.N, -1, self.height_out * self.width_out)

        bottom_diff = col2img(pool_diff, self.H, self.W, self.kernel_size, self.C, 0, self.stride)

        self.backward_time = time.time() - start_time
        return bottom_diff

    def backward_raw_book(self, top_diff):
        start_time = time.time()
        bottom_diff = np.zeros(self.input.shape)
        for idxn in range(top_diff.shape[0]):
            for idxc in range(top_diff.shape[1]):
                for idxh in range(top_diff.shape[2]):
                    for idxw in range(top_diff.shape[3]):
                        # TODO: 最大池化层的反向传播， 计算池化窗口中最大值位置， 并传递损失
                        max_index = np.argwhere(self.max_index[idxn, idxc, 
                                                idxh * self.stride : idxh * self.stride + self.kernel_size, 
                                                idxw * self.stride : idxw * self.stride + self.kernel_size] == 1)[0]
                        # 可能有多个等于1的位置，按顺序，则取第一个

                        bottom_diff[idxn, idxc, idxh * self.stride + max_index[0], idxw * self.stride + max_index[1]] = top_diff[idxn, idxc, idxh, idxw]

        self.backward_time = time.time() - start_time
        return bottom_diff

class FlattenLayer(object):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        assert np.prod(self.input_shape) == np.prod(self.output_shape)
        print('\tFlatten layer with input shape %s, output shape %s.' % (str(self.input_shape), str(self.output_shape)))
    def forward(self, input):
        assert list(input.shape[1:]) == list(self.input_shape)
        # matconvnet feature map dim: [N, height, width, channel]
        # ours feature map dim: [N, channel, height, width]
        self.input = np.transpose(input, [0, 2, 3, 1])
        self.output = self.input.reshape([self.input.shape[0]] + list(self.output_shape))
        return self.output
    def backward(self, top_diff):
        assert list(top_diff.shape[1:]) == list(self.output_shape)
        top_diff = np.transpose(top_diff, [0, 3, 1, 2])
        bottom_diff = top_diff.reshape([top_diff.shape[0]] + list(self.input_shape))
        return bottom_diff
