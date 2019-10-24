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

struct TransformNet
    conv1_w; conv1_b
    conv2_w; conv2_b
    conv3_w; conv3_b
    res4_conv1_w; res4_conv1_b
    res4_conv2_w; res4_conv2_b
    res5_conv1_w; res5_conv1_b
    res5_conv2_w; res5_conv2_b
    res6_conv1_w; res6_conv1_b
    res6_conv2_w; res6_conv2_b
    res7_conv1_w; res7_conv1_b
    res7_conv2_w; res7_conv2_b
    res8_conv1_w; res8_conv1_b
    res8_conv2_w; res8_conv2_b
    dconv9_w; dconv9_b
    dconv10_w; dconv10_b
    conv11_w; conv11_b
    moments
end

function resnet(x, c1w, c1b, c2w, c2b, moments; training=false)
    x0 = x[3:end-2,3:end-2,:,:]
    x1 = conv4(c1w, x) .+ c1b
    x1 = batchnorm(x1, moments[1]; training=training)
    x1 = relu.(x1)
    x1 = conv4(c2w, x1) .+ c2b
    x1 = batchnorm(x1, moments[2]; training=training)
    x = x0 .+ x1
end

function (nn::TransformNet)(x; training=false)
    # check input
    if ndims(x) == 3
        x = reshape(x,size(x)...,1)
    end
    @assert ndims(x) == 4
    h, w, d, n = size(x)
    @assert h == 256
    @assert w == 256
    @assert d == 3
    x = reflect_pad(x, 40)
    # conv1: 336x336x3 -> 336x336x32
    x = reflect_pad(x,4)
    x = conv4(nn.conv1_w, x, stride=1, padding=0) .+ nn.conv1_b
    x = batchnorm(x, nn.moments[1]; training=training)
    x = relu.(x)
    # conv2: 336x336x32 -> 168x168x64
    x = reflect_pad(x,1)
    x = conv4(nn.conv2_w, x, stride=2, padding=0) .+ nn.conv2_b
    x = batchnorm(x, nn.moments[2]; training=training)
    x = relu.(x)
    # conv3: 168x168x64 -> 84x84x128
    x = reflect_pad(x,1)
    x = conv4(nn.conv3_w, x, stride=2, padding=0) .+ nn.conv3_b
    x = batchnorm(x, nn.moments[3]; training=training)
    x = relu.(x)
    # res4: 84x84x128 -> 80x80x128
    x = resnet(x, nn.res4_conv1_w, nn.res4_conv1_b, nn.res4_conv2_w, nn.res4_conv2_b, nn.moments[4:5]; training=training)
    # res5: 80x80x128 -> 76x76x128
    x = resnet(x, nn.res5_conv1_w, nn.res5_conv1_b, nn.res5_conv2_w, nn.res5_conv2_b, nn.moments[6:7]; training=training)
    # res6: 76x76x128 -> 72x72x128
    x = resnet(x, nn.res6_conv1_w, nn.res6_conv1_b, nn.res6_conv2_w, nn.res6_conv2_b, nn.moments[8:9]; training=training)
    # res7: 72x72x128 -> 68x68x128
    x = resnet(x, nn.res7_conv1_w, nn.res7_conv1_b, nn.res7_conv2_w, nn.res7_conv2_b, nn.moments[10:11]; training=training)
    # res8: 68x68x128 -> 64x64x128
    x = resnet(x, nn.res8_conv1_w, nn.res8_conv1_b, nn.res8_conv2_w, nn.res8_conv2_b, nn.moments[12:13]; training=training)
    # conv9: 64x64x128 -> 128x128x64
    x = deconv4(nn.dconv9_w, x, stride=2, padding=1) .+ nn.dconv9_b
    x = cat(x, x[:,end:end,:,:], dims=2)
    x = cat(x, x[end:end,:,:,:], dims=1)
    x = batchnorm(x, nn.moments[14]; training=training)
    x = relu.(x)
    # conv10: 128x128x64 -> 256x256x32
    x = deconv4(nn.dconv10_w, x, stride=2, padding=1) .+ nn.dconv10_b
    x = cat(x, x[:,end:end,:,:], dims=2)
    x = cat(x, x[end:end,:,:,:], dims=1)
    x = batchnorm(x, nn.moments[15]; training=training)
    x = relu.(x)
    # conv11: 256x256x32 -> 256x256x3
    x = reflect_pad(x,4)
    x = conv4(nn.conv11_w, x, stride=1, padding=0) .+ nn.conv11_b
    x = batchnorm(x, nn.moments[16]; training=training)
    x = relu.(x)
    # done
    return x
end

function TransformNet()
    TransformNet(
        # conv1
        param(9,9,3,32), param0(1,1,32,1),
        # conv2
        param(3,3,32,64), param0(1,1,64,1),
        # conv3
        param(3,3,64,128), param0(1,1,128,1),
        # res4
        param(3,3,128,128), param0(1,1,128,1),
        param(3,3,128,128), param0(1,1,128,1),
        # res5
        param(3,3,128,128), param0(1,1,128,1),
        param(3,3,128,128), param0(1,1,128,1),
        # res6
        param(3,3,128,128), param0(1,1,128,1),
        param(3,3,128,128), param0(1,1,128,1),
        # res7
        param(3,3,128,128), param0(1,1,128,1),
        param(3,3,128,128), param0(1,1,128,1),
        # res8
        param(3,3,128,128), param0(1,1,128,1),
        param(3,3,128,128), param0(1,1,128,1),
        # dconv9
        param(3,3,64,128), param0(1,1,64,1),
        # dconv10
        param(3,3,32,64), param0(1,1,32,1),
        # conv11
        param(9,9,32,3), param0(1,1,3,1),
        # moments
        [bnmoments() for _ in 1:16],
    )
end

struct LossNet
    transform_net::TransformNet
    content_layer
    style_layers
    content_target
    style_targets
end

function LossNet(content_img, style_img;
                 content_layer = :conv5_2,
                 style_layers = [:conv1_1, :conv2_1, :conv3_1, :conv4_1, :conv5_1])
    # Preprocess images
    content_img = preproc_img(content_img)
    style_img = preproc_img(style_img)
    
    # Compute targets
    sd = extract_features(style_img)
    cd = extract_features(content_img)
    style_targets = [gram_matrix(sd[l]) for l in style_layers]
    content_target = cd[content_layer]
    
    # Create model
    LossNet(
        TransformNet(),
        content_layer,
        style_layers,
        content_target,
        style_targets
    )
end

struct LossNetIterator
    model::LossNet
    training_images
    iterations
    content_weight
    style_weight
end

function (m::LossNet)(training_images;
                      iterations = 5000,
                      content_weight = 0.001f0,
                      style_weight = 0.0005f0)
    training_images = [
        preproc_img(img)
        for img in training_images
    ]
    LossNetIterator(
        m,
        training_images,
        iterations,
        content_weight,
        style_weight,
    )
end

function Base.iterate(it::LossNetIterator)
    # compute loss
    l = mean(loss(img; model=it.model, content_weight=it.content_weight, style_weight=it.style_weight) for img in it.training_images)
    return (l, ())
end

function Base.iterate(it::LossNetIterator, state)
    # training
    for _ in progress(1:it.iterations)
        for image in it.training_images
            g, l = loss_grad(image; model=it.model, content_weight=it.content_weight, style_weight=it.style_weight)
            update!(image, g)
        end
    end
    # compute loss
    l = mean(loss(img; model=it.model, content_weight=it.content_weight, style_weight=it.style_weight) for img in it.training_images)
    return (l, ())
end

Base.IteratorSize(it::LossNetIterator) = Base.IsInfinite()

function loss(input; model, style_weight, content_weight)
    output = lossnet.transform_net(input)
    # Extract features
    od = extract_features(output)
    style_outputs = [gram_matrix(od[l]) for l in model.style_layers]
    content_output = od[model.content_layer]

    # Compute Loss
    style_loss = mean(mean((so .- st) .^ 2) for (so, st) in zip(style_outputs, model.style_targets)) * 0.25f0
    content_loss = mean((content_output .- model.content_target) .^ 2) * 0.5f0

    # Return
    style_loss * style_weight + content_loss * content_weight
end

function __init__()
    global VGGMODEL = load_model(VGG16)
    global loss_grad = gradloss(loss)
end

function extract_features(x)
    if ndims(x) == 3
        x = reshape(x, size(x)..., 1)
    end
    Dict(takewhile(t -> t[1] != :fc6, VGGMODEL(x)))
end

preproc_img(x) = preprocess_image(VGGMODEL, x)
postproc_img(x) = postprocess_image(VGGMODEL, x)

end
