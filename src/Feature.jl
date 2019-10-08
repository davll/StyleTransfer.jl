module Feature

export FeatureExtractor

using Knet
using VGG

struct FeatureExtractor
    conv1_1_w
    conv1_1_b
    conv1_2_w
    conv1_2_b
    conv2_1_w
    conv2_1_b
    conv2_2_w
    conv2_2_b
    conv3_1_w
    conv3_1_b
    conv3_2_w
    conv3_2_b
    conv3_3_w
    conv3_3_b
    conv3_4_w
    conv3_4_b
    conv4_1_w
    conv4_1_b
    conv4_2_w
    conv4_2_b
    conv4_3_w
    conv4_3_b
    conv4_4_w
    conv4_4_b
    conv5_1_w
    conv5_1_b
    conv5_2_w
    conv5_2_b
    conv5_3_w
    conv5_3_b
    conv5_4_w
    conv5_4_b
    mean_color
end

function FeatureExtractor()
    vgg = load_model(VGG19)
    FeatureExtractor(
        vgg.conv1_1_w,
        vgg.conv1_1_b,
        vgg.conv1_2_w,
        vgg.conv1_2_b,
        vgg.conv2_1_w,
        vgg.conv2_1_b,
        vgg.conv2_2_w,
        vgg.conv2_2_b,
        vgg.conv3_1_w,
        vgg.conv3_1_b,
        vgg.conv3_2_w,
        vgg.conv3_2_b,
        vgg.conv3_3_w,
        vgg.conv3_3_b,
        vgg.conv3_4_w,
        vgg.conv3_4_b,
        vgg.conv4_1_w,
        vgg.conv4_1_b,
        vgg.conv4_2_w,
        vgg.conv4_2_b,
        vgg.conv4_3_w,
        vgg.conv4_3_b,
        vgg.conv4_4_w,
        vgg.conv4_4_b,
        vgg.conv5_1_w,
        vgg.conv5_1_b,
        vgg.conv5_2_w,
        vgg.conv5_2_b,
        vgg.conv5_3_w,
        vgg.conv5_3_b,
        vgg.conv5_4_w,
        vgg.conv5_4_b,
        vgg.mean_color,
    )
end

struct FeatureExtractorIterator
    nn::FeatureExtractor
    x0
end

function (nn::FeatureExtractor)(x)
    # check input
    @assert ndims(x) == 4
    w, h, d, n = size(x)
    @assert d == 3
    # return iterator
    return FeatureExtractorIterator(nn, x)
end

function (nn::FeatureExtractor)(x, layers...)
    Iterators.filter(r -> (r[1] in layers), nn(x)) .|> a -> a[2]
end

function Base.iterate(it::FeatureExtractorIterator)
    nn = it.nn
    x = it.x0
    n = size(x)[end]
    # conv1_1
    x = conv4(nn.conv1_1_w, x; padding=1, stride=1, mode=0) .+ nn.conv1_1_b
    x = relu.(x)
    # return first element and the new state
    ((:conv1_1, x), (:conv1_2, x))
end

function Base.iterate(it::FeatureExtractorIterator, state)
    i, x = state
    nn = it.nn
    n = size(x)[end]
    if i == :conv1_2
        # conv1_2
        x = conv4(nn.conv1_2_w, x; padding=1, stride=1, mode=0) .+ nn.conv1_2_b
        x = relu.(x)
        return ((:conv1_2, x), (:pool1, x))
    elseif i == :pool1
        # pool1
        x = pool(x; window=2, padding=0, stride=2, mode=0)
        return ((:pool1, x), (:conv2_1, x))
    elseif i == :conv2_1
        # conv2_1
        x = conv4(nn.conv2_1_w, x; padding=1, stride=1, mode=0) .+ nn.conv2_1_b
        x = relu.(x)
        return ((:conv2_1, x), (:conv2_2, x))
    elseif i == :conv2_2
        # conv2_2
        x = conv4(nn.conv2_2_w, x; padding=1, stride=1, mode=0) .+ nn.conv2_2_b
        x = relu.(x)
        return ((:conv2_2, x), (:pool2, x))
    elseif i == :pool2
        # pool2
        x = pool(x; window=2, padding=0, stride=2, mode=0)
        return ((:pool2, x), (:conv3_1, x))
    elseif i == :conv3_1
        # conv3_1
        x = conv4(nn.conv3_1_w, x; padding=1, stride=1, mode=0) .+ nn.conv3_1_b
        x = relu.(x)
        return ((:conv3_1, x), (:conv3_2, x))
    elseif i == :conv3_2
        # conv3_2
        x = conv4(nn.conv3_2_w, x; padding=1, stride=1, mode=0) .+ nn.conv3_2_b
        x = relu.(x)
        return ((:conv3_2, x), (:conv3_3, x))
    elseif i == :conv3_3
        # conv3_3
        x = conv4(nn.conv3_3_w, x; padding=1, stride=1, mode=0) .+ nn.conv3_3_b
        x = relu.(x)
        return ((:conv3_3, x), (:conv3_4, x))
    elseif i == :conv3_4
        # conv3_4
        x = conv4(nn.conv3_4_w, x; padding=1, stride=1, mode=0) .+ nn.conv3_4_b
        x = relu.(x)
        return ((:conv3_4, x), (:pool3, x))
    elseif i == :pool3
        # pool3
        x = pool(x; window=2, padding=0, stride=2, mode=0)
        return ((:pool3, x), (:conv4_1, x))
    elseif i == :conv4_1
        # conv4_1
        x = conv4(nn.conv4_1_w, x; padding=1, stride=1, mode=0) .+ nn.conv4_1_b
        x = relu.(x)
        return ((:conv4_1, x), (:conv4_2, x))
    elseif i == :conv4_2
        # conv4_2
        x = conv4(nn.conv4_2_w, x; padding=1, stride=1, mode=0) .+ nn.conv4_2_b
        x = relu.(x)
        return ((:conv4_2, x), (:conv4_3, x))
    elseif i == :conv4_3
        # conv4_3
        x = conv4(nn.conv4_3_w, x; padding=1, stride=1, mode=0) .+ nn.conv4_3_b
        x = relu.(x)
        return ((:conv4_3, x), (:conv4_4, x))
    elseif i == :conv4_4
        # conv4_4
        x = conv4(nn.conv4_4_w, x; padding=1, stride=1, mode=0) .+ nn.conv4_4_b
        x = relu.(x)
        return ((:conv4_4, x), (:pool4, x))
    elseif i == :pool4
        # pool4
        x = pool(x; window=2, padding=0, stride=2, mode=0)
        return ((:pool4, x), (:conv5_1, x))
    elseif i == :conv5_1
        # conv5_1
        x = conv4(nn.conv5_1_w, x; padding=1, stride=1, mode=0) .+ nn.conv5_1_b
        x = relu.(x)
        return ((:conv5_1, x), (:conv5_2, x))
    elseif i == :conv5_2
        # conv5_2
        x = conv4(nn.conv5_2_w, x; padding=1, stride=1, mode=0) .+ nn.conv5_2_b
        x = relu.(x)
        return ((:conv5_2, x), (:conv5_3, x))
    elseif i == :conv5_3
        # conv5_3
        x = conv4(nn.conv5_3_w, x; padding=1, stride=1, mode=0) .+ nn.conv5_3_b
        x = relu.(x)
        return ((:conv5_3, x), (:conv5_4, x))
    elseif i == :conv5_4
        # conv5_4
        x = conv4(nn.conv5_4_w, x; padding=1, stride=1, mode=0) .+ nn.conv5_4_b
        x = relu.(x)
        return ((:conv5_4, x), (:pool5, x))
    elseif i == :pool5
        # pool5
        x = pool(x; window=2, padding=0, stride=2, mode=0)
        return ((:pool5, x), (:fc6, x))
    else
        return nothing
    end
end

Base.length(it::FeatureExtractorIterator) = 21

end
