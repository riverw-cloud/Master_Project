function irm_grid(data) 
    % combine the point with same coordinates and add IRI
    unique_coords = unique(data(:, 1:2), 'rows');
    num_unique_coords = size(unique_coords, 1);
    combined_data = zeros(num_unique_coords, 3);
    
    for i = 1:num_unique_coords
        x = unique_coords(i, 1);
        y = unique_coords(i, 2);
        combined_data(i, 1) = x;
        combined_data(i, 2) = y;
        combined_data(i, 3) = sum(data(data(:, 1) == x & data(:, 2) == y, 5));
    end
    
    % set range and resolution
    gridSize = 0.0523;
    iriMax = max(combined_data(:, 3));
    iriMin = min(combined_data(:, 3));

    
    
    
    figure;
    colormap(parula);
    
    for i = 1:num_unique_coords
        x = combined_data(i, 1);
        y = combined_data(i, 2);
        iri = combined_data(i, 3);
        normalizedIri = (iri - iriMin) / (iriMax - iriMin);
        % find boundary
        xMin = x - gridSize / 2;
        xMax = x + gridSize / 2;
        yMin = y - gridSize / 2;
        yMax = y + gridSize / 2;
        color = colormap(parula);
        colorIndex = round(normalizedIri * (size(color, 1) - 1)) + 1;
        % draw the grid
        rectangle('Position', [xMin, yMin, gridSize, gridSize], 'FaceColor',color(colorIndex, :));
        hold on;
    end
    
    colorbar;
    % draw the red sector
    for i = 1:length(data)
        
        x = data(i, 1);
        y = data(i, 2);
 
        
        
        x1 = x ;
        y1 = y ;
        x2 = x + 0.25 * gridSize * data(i, 3);
        y2 = y + 0.25 * gridSize * data(i, 4);
        

        plot([x1, x2], [y1, y2], 'r');
    end
    

    axis equal;
    xlabel('X');
    ylabel('Y');
    
   
    
    
    
    hold off;
end