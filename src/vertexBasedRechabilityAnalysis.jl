module vertexBasedRechabilityAnalysis

    using LazySets, LinearAlgebra
    using JuMP
    using Gurobi

    using Combinatorics
    using MinimumVolumeEllipsoids
    using MultivariateStats
    using Flux: params

    include("utils/utils.jl")

    export affine_mapping, zeros_verification, get_array_position, get_points_per_orthant, remove_empty_orthants, merging_sets, check_inclusion

    include("utils/vertexOperations.jl")

    export identify_adjascent_vertices, identifying_orthant_intersection_points, comput_internal_intersections, compute_intersections, filtering_zeros, convert_to_matrix, convert_to_vector, identify_non_vertices, elliptic_envelop

    include("utils/origin.jl")

    export origin_search

    export network_mapping, network_mapping2, network_mapping3

    function network_mapping(P_cp, neural_network)
    
        input_dimensionality = size(P_cp)[1];
    
        for i in 1:length(neural_network)-1
    
            P_hat = affine_mapping(P_cp, params(neural_network[i])[1], params(neural_network[i])[2]);
            P_hat = compute_intersections(P_hat, input_dimensionality)
    
            P_cp = filtering_zeros(P_hat);
            # P_cp = identify_non_vertices(P_cp);
            P_cp = elliptic_envelop(P_cp)
    
        end
    
        P_hat = affine_mapping(P_cp, params(neural_network[length(neural_network)])[1], params(neural_network[length(neural_network)])[2]);
    
        return P_hat
    
    end;

    function network_mapping2(P_cp, neural_network)
    
        input_dimensionality = size(P_cp)[1];

        P_input = [P_cp]
    
        for j in 1:length(neural_network)
    
            P_hat_div_aux = []
            P_size = size(P_input)[1]
    
            for i in 1:P_size
                
                P_hat = copy(P_input[i])
    
                if !isempty(P_hat)
    
                    if j > 1
    
                        P_hat = filtering_zeros(P_hat)
    
                        if length(size(P_hat)) > 1
    
                            P_hat = identify_non_vertices(P_hat)
    
                        end
    
                    end
    
                    P_hat = affine_mapping(P_hat, params(neural_network[j])[1], params(neural_network[j])[2])
    
                    if j < length(neural_network)
    
                        if length(size(P_hat)) > 1
    
                            P_hat = compute_intersections(P_hat, input_dimensionality)
    
                        end
    
                        P_hat = get_points_per_orthant(P_hat);
                        P_hat = remove_empty_orthants(P_hat);
    
                        if isempty(P_hat_div_aux)

                            P_hat_div_aux = P_hat

                        else

                            append!(P_hat_div_aux, P_hat)

                        end
    
                    else
    
                        push!(P_hat_div_aux, P_hat)
    
                    end
    
                end
    
            end
    
            P_input = copy(P_hat_div_aux)
    
        end
        
        return P_input 
    
    end;

    function network_mapping3(P_cp, neural_network, divs)
    
        input_dimensionality = size(P_cp)[1];
    
        P_input = [P_cp]
    
        for j in 1:length(neural_network)
    
            input_dims = size(params(neural_network.layers[j])[1])[2];
            output_dims = size(params(neural_network.layers[j])[1])[1];
    
            P_hat_div_aux = []
    
            for i in 1:size(P_input)[1]
    
                if !isempty(P_input[i])
    
                    P_hat = copy(P_input[i])
    
                    if j > 1
    
                        P_hat = filtering_zeros(P_hat)
    
                        if length(size(P_hat)) > 1
    
                            P_hat = identify_non_vertices(P_hat)
    
                        end
    
                    end
    
                    P_hat = affine_mapping(P_hat, params(neural_network[j])[1], params(neural_network[j])[2])
    
                    if j < length(neural_network)
    
                        if length(size(P_hat)) > 1
    
                            P_hat = compute_intersections(P_hat, input_dimensionality);
    
                        end
    
                        P_hat = get_points_per_orthant(P_hat);
                        P_hat = remove_empty_orthants(P_hat);
                        P_hat = merging_sets(P_hat, divs);
    
                        if isempty(P_hat_div_aux)
    
                            P_hat_div_aux = P_hat
    
                        else
    
                            append!(P_hat_div_aux, P_hat)
    
                        end
    
    
                    else
    
                        push!(P_hat_div_aux, P_hat)
    
                    end
    
                end
    
            end
    
            P_input = copy(P_hat_div_aux)
    
        end
    
        return P_input 
    
    end; 
    
end;