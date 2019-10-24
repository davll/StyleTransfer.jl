module Utils

export gram_matrix
export reflect_pad

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

function reflect_pad(x, padding)
    x = cat(x[:,padding:-1:1,:,:], x, x[:,end:-1:end-padding+1,:,:], dims=2)
    x = cat(x[padding:-1:1,:,:,:], x, x[end:-1:end-padding+1,:,:,:], dims=1)
    return x
end

end
