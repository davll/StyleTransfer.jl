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

function loss(input; content_target, style_targets, content_weight = 1.0f0, style_weight = 2.0f0)
    x = input .* 255 .- Utils.MEAN_COLOR
    outputs = extract_features(x, STYLE_LAYERS..., CONTENT_LAYER)
    style_outputs = outputs[1:end-1] .|> gram_matrix
    content_output = outputs[end]
    style_loss = mean(mean((so .- st) .^ 2) for (so, st) in zip(style_outputs, style_targets))
    content_loss = mean((content_output .- content_target) .^ 2)
    style_loss * style_weight + content_loss * content_weight
end

loss_grad = gradloss(loss)

function style_transfer(content_img, style_img)
    # Preprocess images
    content_img = preprocess_image(content_img);
    style_img = preprocess_image(style_img);

    # Compute targets
    style_targets = extract_features(style_img, STYLE_LAYERS...) .|> gram_matrix;
    content_target = extract_features(content_img, CONTENT_LAYER)[1];

    # Variable
    image = Param(content_img, Adam())

    # Training
    for _ in progress(1:5000)
        g, l = loss_grad(image; content_target=content_target, style_targets=style_targets)
        update!(image, g)
    end

    # Output
    img = permutedims(value(image), (3,1,2))
    return colorview(RGB, img)
end

end
