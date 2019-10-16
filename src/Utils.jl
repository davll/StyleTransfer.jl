module Utils

export extract_features
export preprocess_image
export postprocess_image
export gram_matrix

using Knet
using Images
using ..Feature

function extract_features(x, layers...)
    @assert ndims(x) == 3
    x = reshape(x, size(x)..., 1)
    FEATURE_EXTRACTOR(x, layers...)
end

function preprocess_image(img)
    x = channelview(img) .|> Float32
    x = permutedims(x, (2,3,1))
    x = x .* 255 .- MEAN_COLOR
    if Knet.gpu() >= 0
        x = gpucopy(x)
    end
    return x
end

function postprocess_image(x; clamp_pixels=true)
    x = value(x)
    if Knet.gpu() >= 0
        x = cpucopy(x)
    end
    x = (x .+ MEAN_COLOR) ./ 255
    if clamp_pixels
        x = clamp.(x, 0.0f0, 1.0f0)
    end
    x = permutedims(x, (3,1,2))
    return colorview(RGB, x)
end

function gram_matrix(features; normalize=true)
    if ndims(features) == 4
        features = features[:,:,:,1]
    end
    @assert ndims(features) == 3
    H, W, C = size(features)
    feat_reshaped = reshape(features, (H*W, C))
    gram_mat = transpose(feat_reshaped) * feat_reshaped
    @assert size(gram_mat) == (C, C)
    if normalize
        gram_mat ./ (H*W)
    else
        gram_mat
    end
end

function __init__()
    global FEATURE_EXTRACTOR = FeatureExtractor()
    global MEAN_COLOR = FEATURE_EXTRACTOR.mean_color
end

end
