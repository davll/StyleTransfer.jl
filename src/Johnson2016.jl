"""
Perceptual Losses for Real-Time Style Transfer and Super-Resolution

Justin Johnson, Alexandre Alahi, Li Fei-Fei

https://arxiv.org/abs/1603.08155
https://github.com/jcjohnson/fast-neural-style
https://cs.stanford.edu/people/jcjohns/eccv16/
https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf
https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16Supplementary.pdf
https://cs.stanford.edu/people/jcjohns/
"""
module Johnson2016

using Knet
using VGG
using Statistics
using IterTools
using ..Utils

struct ImageTransformNet
    conv1_w; conv1_b
    conv2_w; conv2_b
    conv3_w; conv3_b
    res4::Res
    res5::Res
    res6::Res
    res7::Res
    res8::Res
    dconv9_w; dconv9_b
    dconv10_w; dconv10_b
    dconv11_w; dconv11_b
end

function (nn::ImageTransformNet)(x)
    # check input
    if ndims(x) == 3
        x = reshape(x,size(x)...,1)
    end
    @assert ndims(x) == 4
    h, w, d, n = size(x)
    @assert h == 256 + 80
    @assert w == 256 + 80
    @assert d == 3
    # conv1: 336x336x3 -> 336x336x32
    x = conv4(nn.conv1_w, x, stride=1, padding=2) .+ nn.conv1_b
    x = batchnorm(x)
    x = relu.(x)
    # conv2: 336x336x32 -> 168x168x64
    x = conv4(nn.conv2_w, x, stride=2, padding=1) .+ nn.conv2_b
    x = batchnorm(x)
    x = relu.(x)
    # conv3: 168x168x64 -> 84x84x128
    x = conv4(nn.conv3_w, x, stride=2, padding=1) .+ nn.conv3_b
    x = batchnorm(x)
    x = relu.(x)
    # res4: 84x84x128 -> 80x80x128
    x = nn.res4(x)
    # res5: 80x80x128 -> 76x76x128
    x = nn.res5(x)
    # res6: 76x76x128 -> 72x72x128
    x = nn.res6(x)
    # res7: 72x72x128 -> 68x68x128
    x = nn.res7(x)
    # res8: 68x68x128 -> 64x64x128
    x = nn.res8(x)
    # dconv9: 
    # dconv10
    # dconv11
end

function ImageTransformNet()
    ImageTransformNet(
        param(9,9,3,32), param0(1,1,32,1),
        param(3,3,32,64), param0(1,1,64,1),
        param(3,3,64,128), param0(1,1,128,1),
        #
        #
        #
        #
        #
        
    )
end

struct Res
    conv1_w; conv1_b
    conv2_w; conv2_b
end

function Res()
    Res(
        param(), param0(),
        param(), param0(),
    )
end

function (nn::Res)(x)
    y = x
    x = conv4(nn.conv1_w, x) .+ nn.conv1_b
    x = batchnorm(x)
    x = relu.(x)
    x = conv4(nn.conv2_w, x) .+ nn.conv2_b
    x = batchnorm(x)
    x .+ y
end

end
