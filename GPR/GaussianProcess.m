classdef GaussianProcess < handle
    properties
        % Specify RM or IRM, 0 for RM and 1 for IRM
        map_flag

        % Input map data for experiment
        input_data

        % IRM
        irm_data

        % Number of points in the map
        input_data_length;

        % Projected irm grid
        project_grid;

        % Training set, 3D position in the workspace
        train_X;
        
        % Training set, index
        train_y;
        
        % Test set, 3D position in the workspace
        test_X;
        
        % Grounth truth,  index
        ground_truth_y;
        
        % Predicted  index
        test_y;
        
        % mean of train_X
        mean_train_X;

        % std of train_X
        std_train_X;

        % mean of train_y
        mean_train_y;

        % std of train_y
        std_train_y;

        % Scales of input data size 
        training_scales;

        % Optimal scale
        alpha_star;
        
        % All training set sizes
        training_sizes;
        
        % Scale of input data size
        test_scale;
        
        % Test set size
        test_set_size;

        % GPR model
        gprMdl;
        
        % sample num on outlier
        sub_sample_num;

        % Mahalanobis Distance for each points
        Mi;

        % Mahalanobis Distance for each points after retraining
        Mii;

        % Overall Mahalanobis Distance 
        MI;

        % Overall Mahalanobis Distance after retraining
        MII;
        
        % Outliers
        outlier_1;

        % Outliers after retraining
        outlier_2;
       
        
        % Current training set size 
        training_set_size;

        % Indices of test set
        test_indices;

        % Indices of training set
        training_indices;

        % Flag to optimize the hyperparameter 
        optimization_flag;

        % hyper paramter after cross validation
        optimizedHyperparameters;

        out1;
        out2;
        out3;
    
    end

    methods
        function this = GaussianProcess(input_filename,map_flag)
            if nargin == 2
                this.map_flag = map_flag;
                if(this.map_flag == 0)
                    disp('Loading Reachability Map');
                    this.input_data = load(input_filename).rm;
                    this.input_data_length = length(this.input_data);
                    this.alpha_star = 0.3;
                elseif(this.map_flag == 1)
                    disp('Loading Inverse Reachability Map');
                    this.irm_data = load(input_filename).data;
                    this.input_data = this.irm_data.z_layers{1};
                    this.input_data_length = length(this.input_data.grid);
                    this.alpha_star = 0.25;
                end
                
                
                this.training_sizes = round(this.alpha_star * this.input_data_length);
                this.test_scale = 0.05;
                this.test_set_size = round(this.test_scale*this.input_data_length);    
                this.sub_sample_num = 50;
                rng(1);

                % Preallocate
                this.Mi = zeros(length(this.training_sizes), this.test_set_size);
                this.Mii = zeros(length(this.training_sizes), this.test_set_size);
                this.MI = zeros(length(this.training_sizes), 1);
                this.MII = zeros(length(this.training_sizes), 1);
                this.test_indices = randi(this.input_data_length,1,this.test_set_size);
            else

                error('Please enter a valid file name and map flag');
            end
        end


        function training_indices = generate_training_indices(this)
            % all indices of input data
            all_indices = 1:this.input_data_length;  
            % initial training indices
            initial_training_indices = round(linspace(1, this.input_data_length, this.training_set_size));
           
            % exclude the test indices from training indices
            valid_training_indices = setdiff(initial_training_indices, this.test_indices);
            
            % if num of valid training indices < required training set size, keep
            % generating
            while length(valid_training_indices) < this.training_set_size
                remaining_indices = setdiff(all_indices, this.test_indices);  % exclude the test indices from all indices
                remaining_indices = setdiff(remaining_indices, valid_training_indices);  % exclude the generated indices from remaining indices
                remaining_count = this.training_set_size - length(valid_training_indices); % num of indices need to be generated
                additional_indices = datasample(remaining_indices, remaining_count);  % generate additional indices from the remaining indices
                valid_training_indices = [valid_training_indices, additional_indices];
            end
            valid_training_indices = valid_training_indices(1:this.training_set_size);
        
            valid_training_indices(valid_training_indices < 1) = 1;
            valid_training_indices(valid_training_indices > this.input_data_length) = this.input_data_length;
        
            training_indices = valid_training_indices;
        end

        function training_indices = generate_random_training_indices(this)
            % all indices of input data
            all_indices = randperm(this.input_data_length); 
           
            
            
            % select the indices from random set, until it is enough
            training_indices = [];
            count = 0;
            for i = 1:this.input_data_length
                if ~ismember(all_indices(i), this.test_indices)
                    training_indices = [training_indices; all_indices(i)];
                    count = count + 1;
                end
                if count == this.training_set_size
                    break; 
                end
            end
        end

        function [M,m] = retraining_and_test(this)
  
            indices_on_outlier = [];
            for j = 1:length(this.outlier_1)
                mean_index = this.test_indices(this.outlier_1(j)); % mean of the normal distribution
                subsample_indices_on_outlier = round(normrnd(mean_index, 10, 1, this.sub_sample_num)); % indices around this outlier
                subsample_indices_on_outlier(subsample_indices_on_outlier < 1) = 1;
                subsample_indices_on_outlier(subsample_indices_on_outlier > this.input_data_length) = this.input_data_length;
                indices_on_outlier = [indices_on_outlier,subsample_indices_on_outlier]; 
            end
        
            this.training_indices = [indices_on_outlier,this.training_indices];
            this.training_indices = unique(this.training_indices);
            
            [M,m] = training_and_test(this);
            this.test_y(this.test_y<0) = 0;
            
            
    
        end

        function outlier = outlier_detection(this)
            outlier = [];
            
            for i = 1:length(this.Mi)
                if this.Mi(i) > 3.841
                    outlier = [outlier, i];
                end
            end
            if (isempty(outlier))
                if(this.map_flag == 0)
                    MAD = median(abs(this.Mi - median(this.Mi)));
                    k = 1/sqrt(median(this.Mi));
                    
                    T = median(this.Mi) + k * MAD;
                elseif(this.map_flag == 1)
                    T = 0.7 * max(this.Mi);
                end
                for i = 1:length(this.Mi)
                    if this.Mi(i) > T
                        outlier = [outlier, i];
                    end
                end
            end
            
        end

        function preprocessing(this)
            if(this.map_flag == 0)
                this.train_X = this.input_data(this.training_indices,1:3);
                this.train_y = this.input_data(this.training_indices,4);
                this.test_X = this.input_data(this.test_indices,1:3);
                this.ground_truth_y = this.input_data(this.test_indices,4);
  
            elseif(this.map_flag == 1)
                this.project_grid = [this.input_data.grid(:,1:3),sin(this.input_data.grid(:,4)),cos(this.input_data.grid(:,4))];
                this.train_X = this.project_grid(this.training_indices,:);
                this.train_y = this.input_data.iris(this.training_indices);
                this.test_X = this.project_grid(this.test_indices,:);
                this.ground_truth_y = this.input_data.iris(this.test_indices);
            end
            % normalisation
            this.mean_train_X = mean(this.train_X,1);
            this.std_train_X = std(this.train_X,1);
            this.std_train_X(3) = this.std_train_X(3) + 1e-16;
            this.mean_train_y = mean(this.train_y,1);
            this.std_train_y = std(this.train_y,1);
            this.train_X = (this.train_X - this.mean_train_X)./this.std_train_X;
            this.train_y = (this.train_y - this.mean_train_y)./this.std_train_y;
            this.test_X = (this.test_X - this.mean_train_X)./this.std_train_X;
        end

        function [M,m] = training_and_test(this)
            preprocessing(this);
            if(this.optimization_flag == 0)
                if(~isempty(this.optimizedHyperparameters))
                    this.gprMdl = fitrgp(this.train_X, this.train_y, 'KernelFunction', this.optimizedHyperparameters.KernelFunction, ...
                                                            'KernelParameters', this.optimizedHyperparameters.KernelParameters, ...
                                                            'Sigma', this.optimizedHyperparameters.Sigma,'PredictMethod','exact');
                else
                    this.gprMdl = fitrgp(this.train_X, this.train_y,'KernelFunction','squaredexponential','PredictMethod','exact');
                end
            elseif(this.optimization_flag == 1)
                % use maximise marginal log-likelihood to optimisa the
                % hyperparameter
                this.gprMdl = fitrgp(this.train_X, this.train_y, 'KernelFunction', 'squaredexponential', 'OptimizeHyperparameters', 'auto', ...
                                       'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', ...
                                       'expected-improvement-plus'),'PredictMethod','exact');
                this.optimizedHyperparameters = struct('KernelFunction', this.gprMdl.KernelFunction, 'KernelParameters', this.gprMdl.KernelInformation.KernelParameters, 'Sigma', this.gprMdl.Sigma);
            elseif(this.optimization_flag == 2)
                
                cvp = cvpartition(length(this.training_indices), 'KFold', 5); % 5 partition cross-validation
           
                hyperparameterOpts = struct('Optimizer', 'bayesopt', 'MaxObjectiveEvaluations', 50, 'Verbose', 1, 'CVPartition', cvp);

                % use cross-validation to optimise the hyperparameter
                this.gprMdl = fitrgp(this.train_X, this.train_y, 'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions', hyperparameterOpts,'PredictMethod','exact');
                this.optimizedHyperparameters = struct('KernelFunction', this.gprMdl.KernelFunction, 'KernelParameters', this.gprMdl.KernelInformation.KernelParameters, 'Sigma', this.gprMdl.Sigma);

                
            end
            
            this.optimization_flag = 0;
            %% Get the predicted values
        
            % Note that std is computed on a prediction-by-prediction basis
           
            [this.test_y, Std] = predict(this.gprMdl, this.test_X);
            % denormalisation
            this.train_X = this.train_X .* this.std_train_X + this.mean_train_X;
            this.train_y = this.train_y .* this.std_train_y + this.mean_train_y;
            this.test_X = (this.test_X.*this.std_train_X) + this.mean_train_X;
            this.test_y = (this.test_y.*this.std_train_y) + this.mean_train_y;

            error = this.test_y - this.ground_truth_y;
            
            % Exploit the fact that e^2/P = (e/sqrt(P))^2
            m = error ./ Std;
            m = m.^2;
            M = sqrt(error'*(Std*Std')*error);
            
            
        end

        function visual_gradient(this)
            if(this.map_flag == 0)
                
                coord = this.input_data(:,1:3);
                RI = this.input_data(:,4);
                resolution = 0.0511;
                gradient_mat = map_gradient(resolution,coord,RI);
                
                
                
                figure;
                scatter3(coord(:, 1), coord(:, 2), coord(:, 3), 20, gradient_mat, 'filled');
                colorbar;
                xlabel('X');
                ylabel('Y');
                zlabel('Z');
                title('RM Gradient');
                
                
                hold on;
                threshold = median(gradient_mat) + 1/median(gradient_mat)*median(abs(gradient_mat-median(gradient_mat))); 
                red_indices = find(gradient_mat >= threshold);
                disp(length(red_indices)/this.input_data_length);
                scatter3(coord(red_indices, 1), coord(red_indices, 2), coord(red_indices, 3), 20, 'r', 'filled');
                hold off;
                count = 0;
                for i = 1:length(this.outlier_1)
                    if ismember(this.test_indices(this.outlier_1(i)),red_indices)
                        count = count + 1;
                    end
                end
                disp(count/length(this.outlier_1));
            elseif(this.map_flag == 1)
                
                resolution = 0.0523;
                coord = unique(this.project_grid(:,1:2), 'rows');
                num_unique_coords = size(coord, 1);
                combined_data = zeros(num_unique_coords, 3);
                
                for i = 1:num_unique_coords
                    x = coord(i, 1);
                    y = coord(i, 2);
                    combined_data(i, 1) = x;
                    combined_data(i, 2) = y;
                    combined_data(i, 3) = sum(this.project_grid(this.project_grid(:, 1) == x & this.project_grid(:, 2) == y, 5));
                end
                IRI = combined_data(:,3);
                gradient_mat = map_gradient(resolution,coord,IRI);
                
                
                
                figure;
                scatter(coord(:, 1), coord(:, 2), 20, gradient_mat, 'filled');
                colorbar;
                xlabel('X');
                ylabel('Y');
                title('IRM Gradient');
                
                
                hold on;
                threshold = median(gradient_mat) + 1/median(gradient_mat)*median(abs(gradient_mat-median(gradient_mat))); 
                red_indices = find(gradient_mat >= threshold);

                scatter(coord(red_indices, 1), coord(red_indices, 2), 20, 'r', 'filled');
                scatter(this.project_grid(this.test_indices(this.outlier_1),1),this.project_grid(this.test_indices(this.outlier_1),2),'o', 'LineWidth', 2);
                hold off;
                
            end
            count = 0;
            for i = 1:length(this.outlier_1)
                if ismember(this.test_indices(this.outlier_1(i)),red_indices)
                    count = count + 1;
                end
            end
            disp(count/length(this.outlier_1));

        end

        function test_results_plot(this)
    
            subplot(1,2,1)
            plot(this.Mi)
            title('Original')
            xlabel('Test points')
            ylabel('M distance')
            subplot(1,2,2)
            plot(this.Mii)
            title('Retrained')
            xlabel('Test points')
            ylabel('M distance')
            
        end

        function text_show(this)
            
            fprintf('The indices of outliers are:\n');
            disp(this.test_indices(this.outlier_1))
            fprintf('The M distance of outliers are:\n');
            disp(this.Mi(this.outlier_1));
            fprintf('The M distance of outliers after retraining are:\n');
            disp(this.Mii(this.outlier_1));
            fprintf('The total M distance before retraining = %f\n',this.MI)
            fprintf('The total M distance after retraining = %f\n',this.MII);
            
        end
        
        function save_file(this)
            Mi = this.Mi;
            Mii = this.Mii;
            save('M_distance.mat','Mi');
            save('M_distance_after_retraining','Mii');
        end



        %% Experiments


        function compare_training_size(this)
            this.optimization_flag = 0;
            
            this.training_scales = logspace(-2,log(0.5)/log(10),50);
            this.training_sizes = round(this.training_scales .* this.input_data_length);
            this.test_scale = 0.05;
            this.test_set_size = round(this.test_scale*this.input_data_length);    
            this.MI =  zeros(length(this.training_sizes), 1);
            % Preallocate
            
            this.test_indices = randi(this.input_data_length,1,this.test_set_size);
            for t = 1:length(this.training_sizes)

                this.training_set_size = this.training_sizes(t);
           
                
            
                %% Data pre-processing
                
                %% uniform sample
                this.training_indices = generate_training_indices(this);
               
                %% Training and test
                [this.MI(t), ~] = training_and_test(this);

                %% random sample
                this.training_indices = generate_random_training_indices(this);
               
                %% Training and test
                [this.MII(t), ~] = training_and_test(this);
                disp(t);
              
            end
            plot(this.training_scales,this.MI,'o');
            hold on
            plot(this.training_scales,this.MII,'x');
            ylabel('M distance');
            xlabel('Scale of training size to the dataset size');
            grid on
            title('Mahalanobis Distance over different Training size');
            ylim([0,1.5*max(this.MI)]);
            legend('uniform sampling','random sampling');
        end


        function compare_optimisation_method(this)
            
            this.test_scale = 0.05;
            rng(1);
            this.test_set_size = round(this.test_scale*this.input_data_length);    

            % Preallocate
            
            this.test_indices = randi(this.input_data_length,1,this.test_set_size);

            this.training_set_size = round(0.3*this.input_data_length);
           
            %% Data pre-processing
            this.training_indices = generate_training_indices(this);
            %% Training and test
            this.optimization_flag = 0;
            
            [M,~] = training_and_test(this);
            fprintf('M distance without optimisation = %f\n', M);

            this.optimization_flag = 1;

            [M,~] = training_and_test(this);
            fprintf('M distance with maximise log likelihood optimisation = %f\n', M);

            this.optimization_flag = 2;
            
            [M,~] = training_and_test(this);
            fprintf('M distance with cross-validation optimisation = %f\n', M);
        end

        function evaluate_retraining(this)
            if(this.map_flag == 0)
                this.optimization_flag = 2;
            elseif(this.map_flag == 1)
                this.optimization_flag = 0;
            end
            this.training_set_size = this.training_sizes;

            this.test_scale = 0.05;
            rng(1);
            this.test_set_size = round(this.test_scale*this.input_data_length);    

            % Preallocate
            
            this.test_indices = randi(this.input_data_length,1,this.test_set_size);

       
            
        
            %% Data pre-processing
            
            
            this.training_indices = generate_training_indices(this);
            %% Training and test
            [this.MI, this.Mi] = training_and_test(this);
            %% Outlier detection
            this.outlier_1 = outlier_detection(this);
            
            
            %% Retraining and test
            [this.MII, this.Mii] = retraining_and_test(this);

            text_show(this);
            test_results_plot(this);
            
            visual_gradient(this);


        end


        function visualize(this)
            %%%%%%%%%%%%%%%%
            figure(1)
            this.training_set_size = this.training_sizes;

            this.training_indices = generate_training_indices(this);
            if(this.map_flag == 0)
                scatter3(this.input_data(this.training_indices,1),this.input_data(this.training_indices,2),this.input_data(this.training_indices,3),[],this.input_data(this.training_indices,4),'filled');
                title('original RM')
                cmap = jet(256);
                colormap(flipud(cmap));
        
                colorbar
            elseif(this.map_flag == 1)
                Grid = [this.input_data.grid(:,1:2),sin(this.input_data.grid(:,4)),cos(this.input_data.grid(:,4)),this.input_data.iris];
                irm_grid(Grid);
                title('original IRM')
            end
            
            
            %%%%%%%%%%%%%%%%
            this.optimization_flag = 0;
            
            this.training_set_size = this.training_sizes;

            
            this.training_indices = generate_training_indices(this);
            [~,~] = training_and_test(this);
            
            input1 = (this.project_grid - this.mean_train_X)./this.std_train_X;
            this.out1 = predict(this.gprMdl,input1);
            this.out1 = (this.out1.*this.std_train_y) + this.mean_train_y;
            input1 = input1.*this.std_train_X + this.mean_train_X;
            figure(2)
            
            Grid1 = [input1(:,1:2),input1(:,4:5),this.out1];
            irm_grid(Grid1);
            title('Use 25% points to represent IRM via GPR')
            
            
            %%%%%%%%%%%%%%%%

            this.optimization_flag = 1;

            [~,~] = training_and_test(this);

            input2 = (this.project_grid - this.mean_train_X)./this.std_train_X;
            this.out2 = predict(this.gprMdl,input2);
            this.out2 = (this.out2.*this.std_train_y) + this.mean_train_y;
            input2 = input2.*this.std_train_X + this.mean_train_X;
            figure(3)
            Grid2 = [input2(:,1:2),input2(:,4:5),this.out2];
            irm_grid(Grid2);
            title('Use 25% points to represent RM via GPR with MMLL optimisation')
            

            %%%%%%%%%%%%%%%%%
            [~,~] = retraining_and_test(this);
            input3 = (this.project_grid - this.mean_train_X)./this.std_train_X;
            this.out3 = predict(this.gprMdl,input3);
            this.out3 = (this.out3.*this.std_train_y) + this.mean_train_y;
            input3 = input3.*this.std_train_X + this.mean_train_X;
            this.out3(this.out3<0) = 0; 
            figure(4)
            Grid3 = [input3(:,1:2),input3(:,4:5),this.out3];
            irm_grid(Grid3);
            title('Use 25% points to represent RM via GPR with MMLL optimisation and outlier detection')
           
%         

        end

        function construct_gpr_irm(this)
            num_irm = length(this.irm_data.z_layers);
            gpr_irm = this.irm_data;
            for i = 1:num_irm
                fprintf("Training process:%d/%d\n",i,num_irm);
                this.input_data = this.irm_data.z_layers{i};
                this.input_data_length = length(this.input_data.grid);
                if (this.input_data_length == 0)
                    break;
                end
                
                this.alpha_star = 0.25;

                this.training_sizes = round(this.alpha_star * this.input_data_length);
                this.test_scale = 0.05;
                this.test_set_size = round(this.test_scale*this.input_data_length);    
                this.sub_sample_num = 50;
                rng(1);

                % Preallocate
                this.Mi = zeros(length(this.training_sizes), this.test_set_size);
                this.Mii = zeros(length(this.training_sizes), this.test_set_size);
                this.MI = zeros(length(this.training_sizes), 1);
                this.MII = zeros(length(this.training_sizes), 1);
                this.test_indices = randi(this.input_data_length,1,this.test_set_size);

                evaluate_retraining(this);
                X = (this.project_grid - this.mean_train_X)./this.std_train_X;
                iri = predict(this.gprMdl,X);
                iri = (iri.*this.std_train_y) + this.mean_train_y;
                iri(iri<0) = 0; 
                gpr_irm.z_layers{i}.iris = iri;
                gpr_irm.z_layers{i}.cumsumiris = cumsum(iri);
                data = gpr_irm;
                
            end
            save('gpr_irm_data.mat', 'data','-v7.3');
        end
    end

end