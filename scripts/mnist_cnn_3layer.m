% This script classifies 8x8 MNIST with 5-level CNN implemented on the
% 1T1R memristor array

%% Prepare the dataset
clear all; close all;

% training set resale
load('mnist_training_8x8_0d2.mat');
n_samples = 60000;
xs_trainig = 2*data_0d2(:, 1:n_samples)-0.2;

% trainging set label
ys_training = zeros(10, n_samples);
for i = 1:n_samples
    ys_training(labels(i)+1, i) = 1;
end

% test set rescale
load('mnist_test_8x8_0d2.mat');
n = length(labels);
xs_test = 2*data_0d2(:, 1:n)-0.2;

% test set label
ys_test = zeros(10, n);
for i = 1:n
    ys_test(labels(i)+1, i) = 1;
end

%% Model

% For real array base
load('memristor_1T1R_usable_rows_columns.mat','good_row', 'good_col');
base = multi_array(real_array2_conv(good_row, good_col));

m = model( xbar_v5( base ) );

m.add(unflatten(64, [8 8 1]),                                                             'net_corner', [1 1]);
m.add(conv2D([3 3 1 15],   'strides', [1 1], 'padding', 'same',   'bias_config', [0,0]),  'net_corner', [62 1], 'dp_rep', [2, 1]);
m.add(conv2D([2 2 15 4],   'strides', [1 1], 'padding', 'same',   'bias_config', [0,0]),  'net_corner', [9 16], 'dp_rep', [1, 2]);
m.add(maxpooling2D([2 2],  'strides', [2 2], 'padding', 'valid'),                         'net_corner', [1 1]);
m.add(flatten(),                                                                          'net_corner', [1 1]);
m.add(dense(10, 'activation', 'stable_softmax', 'bias_config', [0,0]),                    'net_corner', [1 24], 'dp_rep', [1, 2]);

%% Auxilary

m.v.DRAW_PLOT = 1;
m.v.DEBUG = 1;

m.summary()
%draw_weights_layout(base);

%% Fit

m.compile(cross_entropy_softmax_loss(),...
    RMSprop('lr', 0.01) , 'draw', 2, 'save', 1);
m.fit( xs_trainig, ys_training, 'batch_size', 100, 'epochs', 2);
    
%% Test-set

ys_testing_hardware = m.predict(xs_test, 'batch_size', 100);
accuracy_testset = m.evaluate( ys_testing_hardware, ys_test);