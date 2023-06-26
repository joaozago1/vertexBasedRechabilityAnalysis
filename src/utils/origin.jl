function origin_search(P_hat, intersection_index_min, intersection_index_max)

    P_hat_aux = nothing

    for j in intersection_index_min:intersection_index_max-1

        for k in j+1:intersection_index_max

            sign_abs_diff = abs.(sign.(P_hat[:,j]) - sign.(P_hat[:,k]))

            if findfirst(==(2), sign_abs_diff) != nothing && size(findall(==(0), round.(P_hat[:,j], digits=6)) ∩ findall(==(0), round.(P_hat[:,k], digits=6)))[1] > 0

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