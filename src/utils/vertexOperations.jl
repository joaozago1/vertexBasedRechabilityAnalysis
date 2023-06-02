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