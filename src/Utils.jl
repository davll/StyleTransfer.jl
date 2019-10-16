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

function postprocess_image(x)
    if Knet.gpu() >= 0
        x = cpucopy(x)
    end
    x = (x .+ MEAN_COLOR) ./ 255
    x = clamp.(x, 0.0f0, 1.0f0)
    x = permutedims(x, (3,1,2))
    return colorview(RGB, x)
end

function gram_matrix(features; normalize=true)
    H, W, C, N = size(features)
    feat_reshaped = reshape(features, (H*W, C))  
    gram_mat = transpose(feat_reshaped) * feat_reshaped  #shape:(C,C)
    if normalize
        return gram_mat ./ (2*H*W*C)
    else
        return gram_mat
    end
end

function __init__()
    global FEATURE_EXTRACTOR = FeatureExtractor()
    global MEAN_COLOR = FEATURE_EXTRACTOR.mean_color
end

end
