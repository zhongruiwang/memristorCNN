%% With Xbar_v5

%% Prepare the dataset
clear; close all;

% train set
% The input to the conv2DLSTM follows
% xs = [64 (actually 8x8), n_sample, time_step (here = 3) ]
load('mnist_conv2DLSTM_training.mat');

xs_training = 0.4 * xs_training - 0.2;
xs_testing = 0.4 * xs_testing - 0.2;

rng(2)

%% Neural network

% For simu array base
% f = @(Vt) 200e-6/1.1*Vt;
% noise = 0.02;
% update_fun = @(G,V,Vt) -G.*(V ~= 0)+(V > 0).* max(f(Vt),G) +...
%     noise.*randn(size(G)).*400e-6; 
% base = multi_array(sim_array1_conv({'random' [128 64] 50e-6 100e-6},...
%     update_fun,0, Inf));

% For real array base
load('May_22_good_row_col.mat','good_row', 'good_col');
base = multi_array(real_array2_conv(good_row, good_col));

m = model( xbar_v5( base ) );

% Gate / RESET voltage
% m.backend.Vg_min = 0.5; m.backend.Vg_max = 1.8; m.backend.Vg0 = 1.15;
m.backend.Vg_min = 0.5; m.backend.Vg_max = 2.4;
m.backend.Vg0 = 1.6; m.backend.V_reset = 1.7; m.backend.V_set = 2.5;
m.backend.ratio_Vg_G = 1.5/100e-6;

% Add layers
m.add( unflatten(64, [8 8 1]),                                                                                                  'net_corner', [1 1]);
m.add( conv2DLSTM([3 3 1 5], [2 2 5 5], 'input_dim', [8 8 1], 'strides_x', [1 1], 'padding_x', 'valid', 'bias_config', [0 0]),  'net_corner', [67 1], 'dp_rep', [1, 2]);
m.add( maxpooling2D([2 2], 'strides', [2 2], 'padding', 'valid'),                                                               'net_corner', [1 1]);
m.add( flatten(),                                                                                                               'net_corner', [1 1]);
m.add( dense(6, 'activation', 'stable_softmax', 'bias_config', [0 0]),                                                          'net_corner', [31 45], 'dp_rep', [1, 2]);

% CLSTM layer size (29*2 rows 20 cols). 
% Dense layer size (45*2 rows 6 cols).
%% Auxilary

m.v.DRAW_PLOT = 1;
m.v.DEBUG = 1;

m.summary()
%draw_weights_layout(base);

no_epoches = 2; % No. of epoches
ys_test_hardware = cell(no_epoches, 1); % Initialize inference results
accuracy = NaN(no_epoches, 3); % Initialize accuracy results

%% Fit

m.compile(cross_entropy_softmax_loss('recurrent_mode', 'last'),...
    RMSprop('lr', 0.01 ), 'draw', 2, 'save', 1);

for epo = 1:no_epoches
    
    % Train
    m.fit(xs_training, ys_training, 'batch_size', 50, 'epochs', 1);
    
    % Inference
    ys_test_hardware{epo} = m.predict(xs_testing, 'batch_size', 50);
    accuracy(epo, 1:3) = m.evaluate(ys_test_hardware{epo}, ys_testing);

end

save('exp_mnistsequence', '-v7.3');