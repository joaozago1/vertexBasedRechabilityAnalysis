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

        H_s_intersection = [findall(==(0), round.(P_intersect[:,i], digits=9)) for i in 1:size(P_intersect)[2]];

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

function compute_intersections(P_hat, input_dimensionality; adj_vertices=nothing)
    
    if adj_vertices === nothing

        adj_vertices = identify_adjascent_vertices(P_hat);

    end

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

    P_hat = round.(P_hat, digits=9);
    P_hat = unique(P_hat, dims=2);

    argsort_aux = sortperm(P_hat[1,:])
    P_hat = P_hat[:,argsort_aux]

    for i in size(P_hat)[2]-1:-1:1

        if norm(P_hat[:,i] - P_hat[:,i+1], Inf) <= 1e-9

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

function calculate_polytope_dim_from_vertices(V)
    m, n = size(V)
    
    if m == 0
        return -1
    elseif m == 1
        return 0 
    end

    v1 = V[1, :]

    differences = V[2:end, :] .- transpose(v1)
    
    return rank(differences, rtol=1e-3)
end

function elliptic_envelop(P_hat_convex_hull)

    dim_P = calculate_polytope_dim_from_vertices(P_hat_convex_hull');
    M = fit(PCA, P_hat_convex_hull; maxoutdim=max(dim_P, 1))

    P_reduced_dim = predict(M, P_hat_convex_hull)

    ϵ = minimum_volume_ellipsoid(P_reduced_dim, 1e-10, 0, 100000, centered=false)

    ellipsoid_center = ϵ.x;
    axis_size = sqrt.(eigvals(inv(ϵ.H ./ 2)))
    eigen_vectors = eigvecs(inv(ϵ.H))

    vertices_hyperrectagle = Matrix{Float64}(undef, length(ellipsoid_center), 0)

    for i in 0:2^length(ellipsoid_center)-1

        bin_aux = digits(i, base=2, pad=length(ellipsoid_center))
        point_aux = ellipsoid_center

        for j in eachindex(ellipsoid_center)

            point_aux += axis_size[j]*eigen_vectors[j,:]*(2*bin_aux[j] - 1)

        end

        vertices_hyperrectagle = hcat(vertices_hyperrectagle, point_aux)

    end

    vertices_hyperrectagle = reconstruct(M, vertices_hyperrectagle)

    return vertices_hyperrectagle

end;

function adj_matrix_to_adj_list(adj_matrix)

    adj_list = Vector{Vector{Int32}}(undef, 0)

    for i in 1:size(adj_matrix, 2)

        push!(adj_list, Vector{Int32}(undef, 0))

        for j in i+1:size(adj_matrix, 1)

            if adj_matrix[i,j] == 1

                push!(adj_list[i], j)

            end

        end

    end

    return adj_list

end;

function compute_hyperrectangle_edges(P)

    adjacency_aux = zeros(size(binary_aux,2), size(binary_aux,2));

    for i in 1:size(P,2)-1

        for j in i+1:size(P,2)

            abs_diff = abs.(P[:,i] - P[:,j])
            null_indices = findall(x -> x == 0, abs_diff)

            if length(null_indices) == 1

                adjacency_aux[i,j] = 1.0

            end

        end

    end

    adjacency_list = adj_matrix_to_adj_list(adjacency_aux);

    return adjacency_list

end