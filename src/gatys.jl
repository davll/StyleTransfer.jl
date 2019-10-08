"""
Original Algorithm

    A Neural Algorithm of Artistic Style
    Leon A. Gatys, Alexander S. Ecker, Matthias Bethge
    2015
    https://arxiv.org/abs/1508.06576

References:
    https://www.tensorflow.org/tutorials/generative/style_transfer
"""
module Gatys

export style_transfer

using Knet
using Statistics
using ..extract_features
using ..gram_matrix
using ..preprocess_image
using ..postprocess_image

STYLE_LAYERS = [:conv1_1, :conv2_1, :conv3_1, :conv4_1, :conv5_1]
CONTENT_LAYER = :conv5_2

function loss_function(content_img, style_img; content_weight, style_weight)
    content_target = extract_features(content_img, CONTENT_LAYER)[1]
    style_targets = extract_features(style_img, STYLE_LAYERS...) .|> gram_matrix
    
    function loss(input)
        outputs = extract_features(preprocess(input), STYLE_LAYERS..., CONTENT_LAYER)
        style_outputs = outputs[1:end-1] .|> gram_matrix
        content_output = outputs[end]
        style_loss = mean(mean((so .- st) .^ 2) for (so, st) in zip(style_outputs, style_targets))
        content_loss = mean((content_output .- content_target) .^ 2)
        style_loss * style_weight + content_loss * content_weight
    end
end

function style_transfer(content_img, style_img; max_iterations=1000, content_weight=1, style_weight=1)
    content_img = preprocess_image(content_img)
    style_img = preprocess_image(style_img)
    
    loss = loss_function(content_img, style_img; content_weight=content_weight, style_weight=style_weight)
    grad = gradloss(loss)
    
    image = Param(content_img, Adam())
    
    for _ in progress(1:max_iterations)
        g, l = grad(image)
        update!(image, g)
    end
    
    postprocess_image(value(image))
end

end
