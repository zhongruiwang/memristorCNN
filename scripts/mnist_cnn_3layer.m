%% CNN for crop and resized MNIST, with Xbar_v7

%% Prepare the dataset
clear; close all;

% train set
xs_train = load('mnist_training_8x8_0d2', 'data_0d2');
xs_train = xs_train.data_0d2;

xs_train = 2 * xs_train - 0.2;

% train labels
ys_train = load('mnist_train_labels_onehot', 'mnist_train_labels');
ys_train = ys_train.mnist_train_labels;

% test set
xs_test = load('mnist_test_8x8_0d2.mat', 'data_0d2');
xs_test = xs_test.data_0d2;

xs_test = 2 * xs_test - 0.2;

% test labels
ys_test = load('mnist_test_labels_onehot', 'mnist_test_labels');
ys_test = ys_test.mnist_test_labels;

rng(2)

%% Model

% For simu array base
% f = @(Vt) 200e-6/1.1*Vt;
% noise = 0.02;
% update_fun = @(G,V,Vt) -G.*(V ~= 0)+(V > 0).* max(f(Vt),G) +...
%     noise.*randn(size(G)).*400e-6; 
% base = multi_array(sim_array1_conv({'random' [123 59] 50e-6 100e-6},...
%     update_fun,0, Inf));

% For real array base
load('May_25_good_row_col.mat','good_row', 'good_col');
base = multi_array(real_array2_conv(good_row, good_col));

m = model( xbar_v7( base ) );

% Gate / RESET voltage
m.backend.Vg_min = 0.5; m.backend.Vg_max = 2.5;
m.backend.Vg0 = 1.6; m.backend.V_reset = 1.7; m.backend.V_set = 2.5;
m.backend.ratio_Vg_G = 1.5/100e-6;

% Add layers
m.add(unflatten(64, [8 8 1]),                                                           'net_corner', [1 1]);
m.add(conv2D([3 3 1 15],  'strides', [1 1],                         'padding', 'same',  'bias_config', [0,0]),  'net_corner', [88 37], 'dp_rep', [2, 1]);
m.add(conv2D([2 2 15 4],  'strides', [1 1], 'activation', 'linear', 'padding', 'same',  'bias_config', [0,0]),  'net_corner', [4 52], 'dp_rep', [1, 2]);
m.add(maxpooling2D([2 2], 'strides', [2 2],                         'padding', 'valid'),                         'net_corner', [1 1]);
m.add(flatten(),                                                                        'net_corner', [1 1]);
m.add(dense(10, 'activation', 'stable_softmax', 'bias_config', [0,0]),                  'net_corner', [38 1], 'dp_rep', [1 2 86]);

%% Auxilary

m.v.DRAW_PLOT = 1;
m.v.DEBUG = 1;

m.summary()
%draw_weights_layout(base);

no_epoches = 2; % No. of epoches
ys_test_hardware = cell(no_epoches, 1); % Initialize inference results
accuracy = NaN(no_epoches, 1); % Initialize accuracy results

%% Fit

m.compile(cross_entropy_softmax_loss(),...
    RMSprop('lr', 0.01) , 'draw', 2, 'save', 1);

for epo = 1:no_epoches
    
    % Train
    m.fit(xs_train, ys_train, 'batch_size', 100, 'epochs', 1);
    
    % Inference
    ys_test_hardware{epo} = m.predict(xs_test, 'batch_size', 100);
    accuracy(epo) = m.evaluate(ys_test_hardware{epo}, ys_test);

end

save('exp_mnist', '-v7.3');