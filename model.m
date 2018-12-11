% Copyright 2018 The Authors. All Rights Reserved.

classdef model < handle
    properties
        layer_list;
        backend;
        
        loss;
        optimizer;
        
        v;
    end
    
    methods
        %%
        function obj = model( backend )
            obj.layer_list = {};
            obj.backend = backend;
            
            obj.v = view();
        end
        %%
        function add( obj, layer, varargin)
            % Add a layer to the neural network model
            % layer (INPUT) : layer object
            % 'net_corner' (Optional input) : 1t1r left upper corner
            % 'dp_ori_hori' (Optional input): true if horizontal diff pair
            
            okargs = { 'net_corner', 'dp_rep'};
            defaults = {[1, 1], [1, 1]};
            [net_corner, dp_rep] = internal.stats.parseArgs(okargs, defaults, varargin{:});
            
            % okargs = { 'net_corner', 'dp_ori_hori'}; % For Xbar_v3 only
            % defaults = {[1 1], 1};
            % [net_corner, dp_ori_hori ] = internal.stats.parseArgs(okargs, defaults, varargin{:});
            
            % In case it's not the first layer
            if ~isempty( obj.layer_list )
                layer.input_dim = obj.layer_list{end}.output_dim;
                
                % RESET weight dimension (due to new input_dim)
                layer.set_weight_dim();
            end
            
            if isnan( layer.input_dim )
                error('Must specify the input dimension for an input layer');
            end
            
            % Backend
            layer.backend = obj.backend;
            
            % Add one layer in the backend ( === Xbar or Xbar_V3 === )
            obj.backend.add_layer(layer.weight_dim, net_corner,...
               numel(obj.layer_list)+1, dp_rep) % Xbar_v5 (all layer)
           
            % obj.backend.add_layer(layer.weight_dim, net_corner,...
            %     numel(obj.layer_list)+1, dp_rep ) % Xbar_v3 (all layer)
            % obj.backend.add_layer(layer.weight_dim, net_corner) % Xbar only (dense/LSTM only)
            
            % Add the current layer to the list
            layer.nlayer = numel(obj.layer_list) + 1;
            obj.layer_list{end+1} = layer;
        end
        %%
        function compile(obj, loss, optimizer, varargin )
            obj.loss = loss;
            
            optimizer.backend = obj.backend;
            obj.optimizer = optimizer;
            
            obj.backend.initialize_weights( varargin{:} );
        end
        %%
        function fit(obj, x_train, y_train, varargin )
            % FIT fit the model according to X_TRAIN and Y_TRAIN
            %   this is the main training function.
            % input:
            %   x_train     Each element in the cell is one training sample
            %               x_train is a 2-dimensional or 3-dimensional
            %               array. x_train( :, n, t)
            %               Will need other structuring for a CNN in the
            %               future.
            %
            %   y_train     The corresponding expected outputs or labels.
            %   epochs
            %   batch_size
            %
            okargs = { 'batch_size', 'epochs'};
            defaults = {10, 1};
            [batch_size, epochs] = internal.stats.parseArgs(okargs, defaults, varargin{:});
            
            %
            if size(x_train, 2) ~= size(y_train, 2)
                error('Training sample number mismatch!');
            end
            
            n_sample = size(x_train, 2);
            
            for ep = 1: epochs
                obj.v.print(['training on ep = ' num2str(ep)]);
                
                % process batch data
                for b_start = 1: batch_size: n_sample
                    b_end = min( b_start + batch_size - 1, n_sample);
                    
                    obj.v.print(['training on ' num2str(b_start) ' to ' num2str(b_end)]);
                    
                    % For simplicity, we only consider the case that all the time series
                    % have the same time duration.
                    % TODO: Test data with various time length.
                    %
                    x_batch = x_train(:, b_start: b_end, :);
                    y_batch = y_train(:, b_start: b_end, :);
                    
                    [loss_value, accuracy] = obj.fit_loop( x_batch, y_batch);
                    obj.v.plot(loss_value, accuracy);
                    display(['Trainning batch accuracy = ' num2str(accuracy)]);
                end
            end
        end
        %%
        function y_test = predict(obj, x_test, varargin)
            okargs = { 'batch_size'};
            defaults = {10};
            [batch_size] = internal.stats.parseArgs(okargs, defaults, varargin{:});        
            
            y_test = [];
            n_sample = size(x_test, 2);
            
            for b_start = 1: batch_size: n_sample
                b_end = min( b_start + batch_size - 1, n_sample);
                
                x_batch = x_test(:, b_start: b_end, :);
                y_test = cat(2, y_test,  obj.forwardpass( x_batch ));
            end
        end
        %%
        function [accuracy, stats] = evaluate( ~, ys, ys_truth, varargin)
            okargs = { 'mode' };
            defaults = { 'simple' };
            [eval_mode] = internal.stats.parseArgs(okargs, defaults, varargin{:});
            
            [~, label_predict] = max(ys );
            [~, label_truth] = max(ys_truth );
            
            accuracy = mean(label_predict == label_truth);
            
            stats = struct();
            
            if strcmp( eval_mode, 'verbose')
                n_class = size(ys, 1);
                stats.win_count = zeros(n_class, n_class);
                
                for i = 1:size(label_predict, 2)
                    stats.win_count( label_truth(1,i, end), label_predict(1,i, end) ) = ...
                        stats.win_count( label_truth(1,i, end), label_predict(1,i, end) ) +1;
                end
                
                
            end
        end
        %%
        function summary(obj)
            
            obj.v.print('------------------------------')
            total = 0;
            for i = 1:length(obj.layer_list)
                
                layer = obj.layer_list{i};
                weights = layer.weight_dim;
                obj.v.print(['layer ' num2str(i) ...
                    ' : input: ' mat2str(layer.input_dim) ...
                    ' output: ' mat2str(layer.output_dim) ...
                    ' weights: ' num2str(weights)]);
                total = total + prod(weights);
            end
            obj.v.print(['parameters: ' num2str(total)])
            obj.v.print('------------------------------')
        end
        %%
        function save(obj, path)
        
            save(path, 'obj')
            obj.v.print(['saved to path' path])
        end       
    end
    
    methods (Access = private)
        %%
        function [loss_value, accuracy] = fit_loop(obj, x_train, y_train)
            % FIT_LOOP an internal function to do the fitting (training)
            % INPUT
            %   x_train:    a two-dimensional array. Each column is one
            %               training sample.
            %   y_train:    Same as x_train.
            %
            ys   = obj.forwardpass( x_train);
            [dys, accuracy]  = obj.loss.calc_delta( ys, y_train );
            loss_value = obj.loss.calc_loss( ys, y_train );
            
            grads = obj.backwardpass( dys );
            obj.optimizer.update( grads );
        end
        %%
        function y_ = forwardpass(obj, x_train)
            % FORWARDPASS the forward pass inferences through all the
            % layers.
            
            y_ = [];
            duration = size(x_train, 3);
            
            for t = 1:duration
                y_time = x_train(:,:,t);
                
                for l = 1:length( obj.layer_list )
                    if t == 1
                        % reset a recurrent layer when starts
                        obj.layer_list{l}.initialize();
                    end
                    
                    y_time = obj.layer_list{l}.call( y_time );
                end
                
                y_ = cat(3, y_, y_time);
            end
        end
        %%
        function grads = backwardpass( obj, dys )
            % BACKWARDPASS calculate the weights gradients
            % INPUT
            %   dys:    final layer delta
            % OUTPUT
            %   grads:  Each element in the cell stores the cumulative
            %           weight gradient for one layer.
            %
            
            grads = cell(1, length( obj.layer_list) );
            grads(:) = {0};
            
            duration = size(dys, 3);
            
            % BP through time
            for t = duration: -1 :1
                % For layers
                dys_time = dys(:,:,t);
                for l = length( obj.layer_list ):-1 :1
                    [gradient, dys_time] = obj.layer_list{l}.calc_gradients(dys_time);
                    grads{l} = grads{l} + gradient;
                end
            end
        end     
    end
end