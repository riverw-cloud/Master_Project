clear;

% If use RM, map_flag = 0
filename1 = 'TCPPBasic-main/+RM/data/arm_stone_reachability_map.mat';

% If use IRM, map_flag = 1
filename2 = 'TCPPBasic-main/+RM/data/as_irm_data.mat';
 
map_flag = 0;

GPR = GaussianProcess(filename1,0);

% Test with varying training sacle
GPR.compare_training_size();

% Test with optimisation method
GPR.compare_optimisation_method();

% Test with outlier detection
GPR.evaluate_retraining();


if (map_flag == 1)
    % Visualise the IRM
    GPR.visualize();

    % Construct the GPR modelled IRM
    GPR.construct_gpr_irm();
end







