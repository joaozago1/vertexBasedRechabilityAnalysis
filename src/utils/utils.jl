function affine_mapping(P,W,b)

    P_hat = W*P .+ b;

    return P_hat;

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