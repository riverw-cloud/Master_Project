function grad = map_gradient(resolution,coordinates,RI)
    epsilon = resolution; % small disturbance
    
    tolerance = 0.001; 
    grad = zeros(size(coordinates, 1), 1); 
    
    % kdtree for NN search
    kd_tree = KDTreeSearcher(coordinates);
    
    % disturb the coordinate and compute gradient
    function compute_gradient(perturbation)
        perturbed_x = point + perturbation;
        [idx, ~] = knnsearch(kd_tree, perturbed_x, 'K', 1);
        
        % make sure the matching point in the tolerance
        if norm(coordinates(idx, :) - perturbed_x) <= tolerance
            % compute gradient
            delta_RI = RI(idx) - ri_value;
            grad_at_d =  delta_RI / sum(perturbation);
        end
    end
    
    for i = 1:size(coordinates, 1)
        point = coordinates(i, :);
        ri_value = RI(i);
        
        grad_at_point = zeros(1,3);
        
        % disturb every dimension
        for j = 1:size(coordinates, 2)
            grad_at_d = 0;
            e = zeros(1,size(coordinates, 2));
            e(j) = 1;
            compute_gradient(epsilon * e); % 
            if(grad_at_d == 0)
                compute_gradient(-epsilon * e); % 
            end
            
            grad_at_point(i) = grad_at_d;
        end
        
        % total gradient
        grad(i) = sqrt(sum(grad_at_point.^2));
    end
end





