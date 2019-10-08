"""
Neural Style Transfer
"""
module StyleTransfer

export extract_features
export MEAN_COLOR
export preprocess_image
export postprocess_image
export gram_matrix

include("Feature.jl")

using Knet
using Images
using .Feature

const FEATURE_EXTRACTOR = FeatureExtractor()
const MEAN_COLOR = FEATURE_EXTRACTOR.mean_color;

function extract_features(img, layers...)
    atype = (Knet.gpu() >= 0 ? KnetArray{Float32} : Array{Float32})
    x = reshape(img .* 255 .- MEAN_COLOR, (size(img)...,1))
    x = convert(atype, x)
    FEATURE_EXTRACTOR(x, layers...)
end

function preprocess_image(img)
    permutedims(channelview(img), (2,3,1)) .|> Float32
end

function postprocess_image(img)
    if Knet.gpu() >= 0
        img = cpucopy(img)
    end
    img = reshape(img, size(img)[1:3]...)
    colorview(RGB, permutedims(clamp.(img, 0.0f0, 1.0f0), (3,1,2)))
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

end # module
