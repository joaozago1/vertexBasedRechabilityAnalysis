module vertexBasedRechabilityAnalysis

    using LazySets, LinearAlgebra
    using JuMP
    using Gurobi

    using Combinatorics
    using Flux: params

    export affine_mapping, identify_adjascent_vertices, identifying_orthant_intersection_points, filtering_zeros, convert_to_matrix, convert_to_vector, identify_non_vertices, network_mapping, origin_search, zeros_verification, get_array_position, get_points_per_orthant, remove_empty_orthants, network_mapping2_removing_non_vertices, merging_sets, network_mapping3, check_inclusion

    function affine_mapping(P,W,b)

        P_hat = W*P .+ b;

        return P_hat;

    end;

    function identify_adjascent_vertices(P_hat)

        adj_vertices = []

        for i in 1:size(P_hat)[2]

            push!(adj_vertices, [])

        end

        Threads.@threads for l in 1:size(P_hat)[2]

            vars = []
            thetas = []

            for i in 1:size(P_hat)[2]

                push!(vars, [])
                push!(thetas, [])

            end

            for j in l+1:size(P_hat)[2]

                model = Model(optimizer_with_attributes(with_optimizer(Gurobi.Optimizer),  "Threads" => 1))
                set_optimizer_attribute(model, "OutputFlag", 0);
                set_optimizer_attribute(model, "LogToConsole", 0);

                push!(vars[j], @variable(model,[i = 1:size(P_hat)[2]]));
                push!(thetas[j], @variable(model, binary=true));

                for i in 1:size(P_hat)[1]

                    @constraint(model, (P_hat[i,l] + P_hat[i,j])/2 .== transpose(P_hat[i,:]) * vars[j][1]);

                end

                @constraint(model, sum(vars[j][1]) == 1);

                @constraint(model, vars[j][1][l] <= thetas[j][1]);
                @constraint(model, vars[j][1][j] <= 1 - thetas[j][1]);

                for i in 1:size(P_hat)[2]

                    @constraint(model, vars[j][1][i] >= 0);

                end

                optimize!(model)

                if termination_status(model) != MOI.OPTIMAL;

                    push!(adj_vertices[l], j)

                end


            end

        end

        return adj_vertices

    end;

    function identifying_orthant_intersection_points(P_hat, adj_vertices)

        P_aux = []

        if length(size(P_hat)) > 1

            for i in 1:size(P_hat)[2]

                push!(P_aux, [])

            end

            Threads.@threads for i in 1:size(P_hat)[2]

                for j in 1:size(adj_vertices[i])[1]

                    A = P_hat[:,i];
                    B = P_hat[:,adj_vertices[i][j]];

                    sign_diff = sign.(A) - sign.(B);

                    if sum(abs.(sign_diff)) != 0 

                        for l in 1:size(A)[1]

                            if sign_diff[l] != 0

                                X_aux = zeros(size(P_hat)[1])

                                lambda = -A[l]/(B[l] - A[l])

                                for k in 1:size(A)[1]

                                    X_aux[k] = (B[k] - A[k])*lambda + A[k]

                                end

                                if isempty(P_aux[i])

                                    P_aux[i] = X_aux

                                else

                                    P_aux[i] = [P_aux[i] X_aux]

                                end

                            end

                        end

                    end

                end

            end

            for i in 1:size(P_aux)[1]

                if !isempty(P_aux[i])

                    P_hat = hcat(P_hat, P_aux[i])

                end

            end

        end

        return P_hat

    end;

    function filtering_zeros(P_hat)

        for i in 1:length(P_hat)    

            if P_hat[i] < 0

                P_hat[i] = 0

            end

        end

        return P_hat

    end;

    function convert_to_matrix(P_cp)

        P_cp2 = []

        for i in 1:size(P_cp)[1]

            if i == 1

                P_cp2 = P_cp[i]

            else

                P_cp2 = [P_cp2 P_cp[i]]

            end

        end

        return P_cp2

    end;

    function convert_to_vector(P_cp)

        aux = []

        for i in 1:size(P_cp)[2]

            push!(aux, Vector(P_cp[:,i]))

        end

        return aux

    end;

    function identify_non_vertices(P_cp)

        model_test = Model(optimizer_with_attributes(with_optimizer(Gurobi.Optimizer),  "Threads" => 1))

        for k in size(P_cp)[2]:-1:1

            set_optimizer_attribute(model_test, "OutputFlag", 0);
            set_optimizer_attribute(model_test, "LogToConsole", 0);

            @variable(model_test, lambda[i = 1:size(P_cp)[2]] >= 0);

            for i in 1:size(P_cp)[1]

                @constraint(model_test, P_cp[i,k] .== transpose(P_cp[i,:]) * lambda);

            end

            @constraint(model_test, sum(lambda) == 1);

            @constraint(model_test, lambda[k] == 0);

            optimize!(model_test);

            if termination_status(model_test) == MOI.OPTIMAL;

                P_cp = P_cp[:, 1:end .!=k]

            end

            empty!(model_test)

        end

        return P_cp

    end;

    function network_mapping(P_cp, rede_neural)

        for i in 1:length(rede_neural)-1

            P_hat = affine_mapping(P_cp, params(rede_neural[i])[1], params(rede_neural[i])[2]);
            adj_vertices = identify_adjascent_vertices(P_hat);

            intersection_index_min = size(P_hat)[2] + 1;
            P_hat = identifying_orthant_intersection_points(P_hat, adj_vertices)
            intersection_index_max = size(P_hat)[2];
            P_hat_aux = origin_search_2(P_hat, intersection_index_min, intersection_index_max)
        
            if P_hat_aux != nothing

                P_hat = [P_hat P_hat_aux];

            end;
            
            P_hat = round.(P_hat, digits=6);

            P_cp = filtering_zeros(P_hat);
            P_cp = identify_non_vertices(P_cp);

        end

        P_hat = affine_mapping(P_cp, params(rede_neural[length(rede_neural)])[1], params(rede_neural[length(rede_neural)])[2]);

        return P_hat

    end;

    function origin_search_2(P_hat, intersection_index_min, intersection_index_max)

        P_hat_aux = nothing

        for j in intersection_index_min:intersection_index_max-1

            for k in j+1:intersection_index_max

                sign_abs_diff = abs.(sign.(P_hat[:,j]) - sign.(P_hat[:,k]))

                if maximum(sign_abs_diff) == 2 && findall(==(0), round.(P_hat[:,j], digits=6))[1] == findall(==(0), round.(P_hat[:,k], digits=6))[1]#&& length(sign_abs_diff[sign_abs_diff .>= 2]) >= input_dims

                    for i in 1:size(P_hat)[1]

                        if sign_abs_diff[i] == 2

                            λ = (0 - P_hat[i,k])/(P_hat[i,j] - P_hat[i,k])

                            aux_point = P_hat[:,j].*λ+P_hat[:,k].*(1-λ)

                            if P_hat_aux == nothing

                                P_hat_aux = aux_point

                            else

                                P_hat_aux = [P_hat_aux aux_point]

                            end

                        end

                    end

                end

            end


        end
        
        if P_hat_aux != nothing

            P_hat_aux = unique(P_hat_aux,dims=2);
            
            
        end
        
        return P_hat_aux
        
    end;

    function origin_search(P_hat)

        model_test = Model(Gurobi.Optimizer);

        set_optimizer_attribute(model_test, "OutputFlag", 0);
        set_optimizer_attribute(model_test, "LogToConsole", 0);

        @variable(model_test, lambda[i = 1:size(P_hat)[2]] >= 0);

        for i in 1:size(P_hat)[1]

            @constraint(model_test, 0 .== transpose(P_hat[i,:]) * lambda);

        end

        @constraint(model_test, sum(lambda) == 1);

        optimize!(model_test);

        if termination_status(model_test) == MOI.OPTIMAL;

            P_hat = [P_hat zeros(size(P_hat)[1])]

        end

        return P_hat

    end;

    function zeros_verification(input_x)

        aux1 = []

        push!(aux1, input_x)

        aux2 = copy(aux1)

        while !isempty(aux2)

            aux2 = []

            for j in 1:size(aux1)[1]

                for i in 1:size(aux1[j])[1]

                    if aux1[j][i] == 0.5

                        aux = copy(aux1[j])
                        aux[i] = 0

                        push!(aux2, aux)

                        aux = copy(aux1[j])
                        aux[i] = 1

                        push!(aux2, aux)

                        break

                    end

                end

            end

            if !isempty(aux2)

                aux1 = copy(aux2)

            end

        end

        return aux1

    end;

    function get_array_position(binary_x)

        count = 1

        for j in 1:size(binary_x)[1]

            count = count + binary_x[j]*2^(j-1)

        end

        return count

    end;

    function get_points_per_orthant(P_hat)
    
        valid_keys = []

        list_dict = Dict()

        if length(size(P_hat)) > 1

            n_points = size(P_hat)[2]

        else

            n_points = 1

        end

        for i in 1:n_points

            binary_positive = (sign.(P_hat[:,i]) .+ 1)./2

            if sum(binary_positive .== 0.5) > 0

                list_binary_intersection = zeros_verification(binary_positive)

                for k in 1:size(list_binary_intersection)[1]

                    pos = get_array_position(list_binary_intersection[k])

                    if !haskey(list_dict, pos)

                        list_dict[pos] = []
                        push!(valid_keys, pos)

                    end

                    if isempty(list_dict[pos])

                        list_dict[pos] = P_hat[:,i]

                    else

                        list_dict[pos] = hcat(list_dict[pos], P_hat[:,i])

                    end

                end

            else

                pos = get_array_position(binary_positive)

                if !haskey(list_dict, pos)

                    list_dict[pos] = []
                    push!(valid_keys, pos)

                end

                if isempty(list_dict[pos])

                    list_dict[pos] = P_hat[:,i]

                else

                    list_dict[pos] = hcat(list_dict[pos], P_hat[:,i])

                end

            end

        end

        list_array = []

        for i in 1:size(valid_keys)[1]

            push!(list_array, list_dict[valid_keys[i]])

        end

        return list_array

    end;


    function remove_empty_orthants(P_hat)

        for i in size(P_hat)[1]:-1:1

            if isempty(P_hat[i])

                deleteat!(P_hat, i)

            end

        end

        return P_hat

    end;

    function network_mapping2(P_cp, rede_neural)

        P_input = [P_cp]

        for j in 1:length(rede_neural)
        
            input_dims = size(params(rede_neural.layers[j])[1])[2];
            output_dims = size(params(rede_neural.layers[j])[1])[1];

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

                    P_hat = affine_mapping(P_hat, params(rede_neural[j])[1], params(rede_neural[j])[2])

                    if j < length(rede_neural)
                    
                        if length(size(P_hat)) > 1

                            adj_vertices = identify_adjascent_vertices(P_hat)
                            intersection_index_min = size(P_hat)[2] + 1;
                            P_hat = identifying_orthant_intersection_points(P_hat, adj_vertices)
                            intersection_index_max = size(P_hat)[2];
                            P_hat_aux = origin_search_2(P_hat, intersection_index_min, intersection_index_max)
                        
                            if P_hat_aux != nothing

                                P_hat = [P_hat P_hat_aux];

                            end;
                            
                            P_hat = round.(P_hat, digits=6);
                        
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

    function merging_sets(P_hat, divs)

        P_aux = []

        for i in 1:Int(ceil(size(P_hat)[1]/divs))

            P_aux2 = []

            for j in (i-1)*divs+1:minimum([(i)*divs, size(P_hat)[1]])

                if isempty(P_aux2)

                    P_aux2 = P_hat[j]

                else

                    P_aux2 = hcat(P_aux2, P_hat[j])

                end

            end

            push!(P_aux, P_aux2)

        end

        return P_aux

    end;

    function network_mapping3(P_cp, rede_neural, divs)

        P_input = [P_cp]

        for j in 1:length(rede_neural)

            input_dims = size(params(rede_neural.layers[j])[1])[2];
            output_dims = size(params(rede_neural.layers[j])[1])[1];

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

                    P_hat = affine_mapping(P_hat, params(rede_neural[j])[1], params(rede_neural[j])[2])

                    if j < length(rede_neural)

                        if length(size(P_hat)) > 1

                            adj_vertices = identify_adjascent_vertices(P_hat)
                            intersection_index_min = size(P_hat)[2] + 1;
                            P_hat = identifying_orthant_intersection_points(P_hat, adj_vertices)
                            intersection_index_max = size(P_hat)[2];
                            P_hat_aux = origin_search_2(P_hat, intersection_index_min, intersection_index_max)
                        
                            if P_hat_aux != nothing

                                P_hat = [P_hat P_hat_aux];

                            end;
                            
                            P_hat = round.(P_hat, digits=6);

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

    function check_inclusion(C,D,R)
    
        for k in 1:size(R)[1]
        
            prod_c = C*R[k]
            
            if ndims(prod_c) == 1
                
                for j in 1:size(prod_c)[1]
    
                    if prod_c[j] > D[j]
    
                        return :violated
    
                    end
    
                end 
    
            else
                
                for i in 1:size(prod_c)[2]
    
                    for j in 1:size(prod_c)[1]
    
                        if prod_c[j,i] > D[j]
    
                            return :violated
    
                        end
    
                    end 
    
                end
                
            end
            
        end
        
        return :holds
    
    end;
    
end;