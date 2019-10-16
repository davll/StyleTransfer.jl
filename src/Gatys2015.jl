"""
A Neural Algorithm of Artistic Style

Leon A. Gatys, Alexander S. Ecker, Matthias Bethge

https://arxiv.org/abs/1508.06576

"""
module Gatys2015

export style_transfer

using Knet
using Images
using Statistics
using ..Utils

STYLE_LAYERS = [:conv1_1, :conv2_1, :conv3_1, :conv4_1, :conv5_1]
CONTENT_LAYER = :conv5_2

function loss(input; content_target, style_targets, content_weight, style_weight)
    outputs = extract_features(input, STYLE_LAYERS..., CONTENT_LAYER)
    style_outputs = outputs[1:end-1] .|> gram_matrix
    content_output = outputs[end]
    style_loss = mean(mean((so .- st) .^ 2) for (so, st) in zip(style_outputs, style_targets)) * 0.25f0
    content_loss = mean((content_output .- content_target) .^ 2) * 0.5f0
    style_loss * style_weight + content_loss * content_weight
end

loss_grad = gradloss(loss)

function style_transfer(content_img, style_img, initial_img=content_img; content_weight = 0.001f0, style_weight = 0.0005f0, max_iterations=5000)
    # Preprocess images
    content_img = preprocess_image(content_img)
    style_img = preprocess_image(style_img)
    initial_img = preprocess_image(initial_img)
    
    # Compute targets
    style_targets = extract_features(style_img, STYLE_LAYERS...) .|> gram_matrix
    content_target = extract_features(content_img, CONTENT_LAYER)[1]

    # Compute loss
    @show loss(initial_img; content_target=content_target, style_targets=style_targets, content_weight=content_weight, style_weight=style_weight)
    
    # Variable
    image = Param(initial_img, Adam())

    # Training
    for _ in progress(1:max_iterations)
        g, l = loss_grad(image; content_target=content_target, style_targets=style_targets, content_weight=content_weight, style_weight=style_weight)
        update!(image, g)
    end

    # Output
    return postprocess_image(image; clamp_pixels=false)
end

end
