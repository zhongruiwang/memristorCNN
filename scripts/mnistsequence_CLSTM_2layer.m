% This script classifies the synthetic MNIST-sequence with 3-level ConvLSTM
% implemented on the 1T1R memristor array

%% Prepare the dataset
clear all; close all;

% The input to the conv2DLSTM follows
% xs = [64 (actually 8x8), n_sample, time_step (here = 3) ]
load('mnistsequence_train_test.mat');

% training set rescale
xs_training = 0.4 * xs_training - 0.2;

% test set rescale
xs_testing = 0.4 * xs_testing - 0.2;

%% Neural network

% For real array base
load('memristor_1T1R_usable_rows_columns.mat','good_row', 'good_col');
base = multi_array(real_array2_conv(good_row, good_col));

m = model(xbar_v5( base ));

m.add( unflatten(64, [8 8 1]),                                                                                                  'net_corner', [1 1]);
m.add( conv2DLSTM([3 3 1 5], [2 2 5 5], 'input_dim', [8 8 1], 'strides_x', [1 1], 'padding_x', 'valid', 'bias_config', [0 0]),  'net_corner', [13 22], 'dp_rep', [2, 1]);
m.add( maxpooling2D([2 2], 'strides', [2 2], 'padding', 'valid'),                                                               'net_corner', [1 1]);
m.add( flatten(),                                                                                                               'net_corner', [1 1]);
m.add( dense(6, 'activation', 'stable_softmax', 'bias_config', [0 0]),                                                          'net_corner', [39 10], 'dp_rep', [1, 2]);

%% Auxilary

m.v.DRAW_PLOT = 1;
m.v.DEBUG = 1;

m.summary
%draw_weights_layout(base);

%% Fit

m.compile(cross_entropy_softmax_loss('recurrent_mode', 'last'),...
    RMSprop('lr', 0.01 ), 'draw', 2, 'save', 1);
m.fit( xs_training(:,sample_sel,:), ys_training(:,sample_sel,:), 'batch_size', 50, 'epochs', 1);

%% Test-set

ys_testing_hardware = m.predict(xs_testing, 'batch_size', 50);
accuracy_testset = m.evaluate( ys_testing_hardware, ys_testing);
