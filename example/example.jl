using Flux
using Flux: onehotbatch, argmax, crossentropy, throttle

using Base.Iterators: repeated

using LazySets

using vertexBasedRechabilityAnalysis

function creating_dataset(N)

    N = parse(Int64, N)

    theta = reshape(LinRange(0,2*pi,N), (1,N));

    r1 = 1;

    x1 = cos.(theta)*r1;
    y1 = sin.(theta)*r1;
    z1 = vcat(ones((1,N)), zeros((1,N)))

    r2 = 3;

    x2 = cos.(theta)*r2;
    y2 = sin.(theta)*r2;
    z2 = vcat(zeros((1,N)), ones((1,N)))

    input_data = hcat(vcat(x1,y1), vcat(x2,y2));
    output_data = hcat(z1, z2);

    return input_data, output_data

end

function training_model(input_data, output_data)

    rede_neural = Chain(
        Dense(2, 4, relu),
        Dense(4, 4, relu),
        Dense(4, 2),
    );
    ps = Flux.params(rede_neural);

    loss(x, y) = Flux.crossentropy(softmax(rede_neural(x)), y)

    evalcb = () -> @show(loss(input_data, output_data))

    datasetx = repeated((input_data, output_data), 5000);

    C = collect(datasetx);

    opt = ADAM();

    Flux.train!(loss, ps, datasetx, opt, cb = throttle(evalcb, 1))

    return rede_neural

end

function verifying_property(rede_neural)

    P = transpose([1.33261 0.0; 1.07811 -0.783289; 0.4118 -1.26739; -0.4118 -1.26739; -1.07811 -0.783289; -1.33261 0.0; -1.07811 0.783289; -0.4118 1.26739; 0.4118 1.26739; 1.07811 0.783289])

    P_hat_exact = network_mapping2(P, rede_neural);

    return check_inclusion([-1 1], [0], P_hat_exact)

end

function main(N)

    input_data, output_data = creating_dataset(N)

    rede_neural = training_model(input_data, output_data)

    @show verifying_property(rede_neural)

end

main(ARGS[1])