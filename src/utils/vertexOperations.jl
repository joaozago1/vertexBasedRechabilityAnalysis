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

            model = Model(optimizer_with_attributes(Gurobi.Optimizer,  "Threads" => 1))
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

                sign_abs_diff = abs.(sign.(A) - sign.(B));

                if findfirst(==(2), sign_abs_diff) != nothing

                    for l in 1:size(A)[1]

                        if sign_abs_diff[l] == 2

                            λ = (0 - A[l])/(B[l] -A[l])

                            X_aux = B .* λ + A .* (1 - λ)

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
        
        P_aux_concat = []

        for i in 1:size(P_aux)[1]

            if !isempty(P_aux[i])
                
                if isempty(P_aux_concat)
                    
                    P_aux_concat = P_aux[i]
                    
                else 

                    P_aux_concat = hcat(P_aux_concat, P_aux[i])
                    
                end

            end

        end

    end

    return P_aux_concat

end;

function comput_internal_intersections(P_intersect)
    
    P_intersect_aux = []

    if length(size(P_intersect)) == 2

        H_s_intersection = [findall(==(0), round.(P_intersect[:,i], digits=6)) for i in 1:size(P_intersect)[2]];

        for j in 1:size(P_intersect)[1]

            index_aux = findall(==([j]), H_s_intersection)

            if !isempty(index_aux)
                
                adj_vertices_aux = identify_adjascent_vertices(P_intersect[:,index_aux])
                P_intersect_temp = identifying_orthant_intersection_points(P_intersect[:,index_aux], adj_vertices_aux)
                
                if length(P_intersect_temp) > 0
                
                    if isempty(P_intersect_aux)

                        P_intersect_aux = P_intersect_temp

                    else

                        P_intersect_aux = [P_intersect_aux P_intersect_temp]

                    end
                    
                end
                
            end

        end

    end
    
    return P_intersect_aux
    
end;

function compute_intersections(P_hat, input_dimensionality)
    
    adj_vertices = identify_adjascent_vertices(P_hat);
    P_intersect = identifying_orthant_intersection_points(P_hat, adj_vertices)
    
    if length(P_intersect) > 0
        
        P_hat = [P_hat P_intersect]
        
        for i in 1:input_dimensionality-1

            P_intersect = comput_internal_intersections(P_intersect)

            if length(P_intersect) > 0

                P_hat = [P_hat P_intersect] 

            end

        end
        
    end

    P_hat = removing_duplicated_vertices(P_hat);

    return P_hat
    
end;

function removing_duplicated_vertices(P_hat)

    P_hat = round.(P_hat, digits=6);
    P_hat = unique(P_hat, dims=2);

    argsort_aux = sortperm(P_hat[1,:])
    P_hat = P_hat[:,argsort_aux]

    for i in size(P_hat)[2]-1:-1:1

        if norm(P_hat[:,i] - P_hat[:,i+1], Inf) <= 1e-6

            P_hat = hcat(P_hat[:,1:i], P_hat[:,i+2:end])

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

    model_test = Model(optimizer_with_attributes(Gurobi.Optimizer,  "Threads" => 1))

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