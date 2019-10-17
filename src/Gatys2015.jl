"""
A Neural Algorithm of Artistic Style

Leon A. Gatys, Alexander S. Ecker, Matthias Bethge

https://arxiv.org/abs/1508.06576

"""
module Gatys2015

export StyleTransfer

using Knet
using VGG
using Statistics
using IterTools
using ..Utils

struct StyleTransfer
    content_layer
    content_target
    style_layers
    style_targets
end

function StyleTransfer(content_img, style_img;
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

    # Create Model
    StyleTransfer(
        content_layer,
        content_target,
        style_layers,
        style_targets,
    )
end

struct StyleTransferIterator
    model::StyleTransfer
    initial_image
    iterations
    content_weight
    style_weight
end

function (self::StyleTransfer)(initial_image;
                               iterations = 5000,
                               content_weight = 0.001f0,
                               style_weight = 0.0005f0)
    StyleTransferIterator(
        self,
        initial_image,
        iterations,
        content_weight,
        style_weight,
    )
end

function Base.iterate(it::StyleTransferIterator)
    # Preprocess input
    img = preproc_img(it.initial_image)
    img_loss = loss(img; model=it.model, content_weight=it.content_weight, style_weight=it.style_weight)
    # Variable
    image = Param(img, Adam())
    # make state
    state = (image,)
    # Return
    Base.iterate(it, state)
end

function Base.iterate(it::StyleTransferIterator, state)
    # decompose state
    (image,) = state
    # Training
    for _ in progress(1:it.iterations)
        g, l = loss_grad(image; model=it.model, content_weight=it.content_weight, style_weight=it.style_weight)
        update!(image, g)
    end
    # compute loss
    img = postproc_img(image)
    img_loss = loss(value(image); model=it.model, content_weight=it.content_weight, style_weight=it.style_weight)
    # make next state
    state = (image,)
    # return
    ((img, img_loss), state)
end

Base.IteratorSize(it::StyleTransferIterator) = Base.IsInfinite()

function loss(input; model, style_weight, content_weight)
    # Extract features
    od = extract_features(input)
    style_outputs = [gram_matrix(od[l]) for l in model.style_layers]
    content_output = od[model.content_layer]

    # Compute Loss
    style_loss = mean(mean((so .- st) .^ 2) for (so, st) in zip(style_outputs, model.style_targets)) * 0.25f0
    content_loss = mean((content_output .- model.content_target) .^ 2) * 0.5f0

    # Return
    style_loss * style_weight + content_loss * content_weight
end

function __init__()
    global VGGMODEL = load_model(VGG19)
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
