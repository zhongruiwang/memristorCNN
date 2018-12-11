classdef dense < layer
    % Fully connected layer class
    
    properties
        nlayer % current layer ID
        backend % backend interface
        
        bias_config % (local) a two dimensional array: [ratio(bias value) rep(use how many inputs)]
        
        % Dimensions
        input_dim
        output_dim
        weight_dim
        
        % Activation func name
        act_name
        
        % (local) input/output histories
        x_in_history
        y_out_history
    end
    
    methods
        function obj = dense( output_dim, varargin )
            % DENSE the construction funciton for a fully connect layer
            
            okargs = {'input_dim', 'activation', 'bias_config'};
            defaults = {NaN, 'linear', [1 1]};
            [obj.input_dim, obj.act_name, obj.bias_config] = internal.stats.parseArgs(okargs, defaults, varargin{:});
            
            obj.output_dim = output_dim;
            
            obj.set_weight_dim();
            obj.initialize();
        end
        %%
        function set_weight_dim(obj)
            % For func dense, set weights dimension
            % Num of rows = output dim
            % Num of col = input dim + 1 (if physical bias)
            
            obj.weight_dim = [obj.output_dim obj.input_dim + obj.bias_config(2) ];
        end
        %%
        function initialize(obj)
            % For func dense, initialize history
            
            obj.x_in_history = [];
            obj.y_out_history = [];
        end
        %%
        function y_out = call(obj, x_in)
            % CALL The forward pass of the layer
            % Input:
            % x_in:   The input of the layer
            % Output:
            % y_out:  The output
            
            n = size(x_in, 2); % The batch size, each input a column
            
            % Add bias to the input.
            %   The bias_config  = [ratio(or bias value)  nrep(use how many inputs)]
            x_in_full = [x_in; repmat( obj.bias_config(1), obj.bias_config(2), n) ];
            
            % Forward propogation
            y_out = obj.backend.multiply( x_in_full, obj.nlayer);
            
            % Activation
            act = activations.get( obj.act_name, 'act' ); % Fetch the func
            y_out = act( y_out );
            
            % Store the output for future backpropogation
            % (note the DIM=3 is because y_out x_in_full are 2D usually)
            obj.y_out_history = cat(3, obj.y_out_history, y_out);
            obj.x_in_history = cat(3, obj.x_in_history, x_in_full);
        end
        
        function [grads, dx] = calc_gradients(obj, dy)
            % CALC_GRADIENTS: The backward pass of the layer
            % Input:
            %   dy:     The delta on this layer
            % Output:
            %   grads:     (dW) The weight gradient in this layer
            %   dx:     The delta for previous layer
            %
            
            % Calculate the delta before activation
            [y_out,  obj.y_out_history] = obj.history_pop( obj.y_out_history);
            [x_in,  obj.x_in_history] = obj.history_pop( obj.x_in_history);
            
            % Activation
            if ~contains( obj.act_name, 'softmax') 
                
                % Fetch activation func derivative func
                act = activations.get( obj.act_name, 'deriv' );
                
                % from deltas (after activation) to nodes before activation
                dy = dy .* act( y_out);
            end
            
            % Calculate the gradient
            grads = dy * x_in.';
            
            % Back propogation (new deltas)
            dx = obj.backend.multiply_reverse( dy, obj.nlayer );
            dx = dx(1: obj.input_dim, :); %???
        end
    end
end