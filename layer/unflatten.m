classdef unflatten < layer
    % Unflatten layer (2D to 4D only)
    
    properties
        
        nlayer % current layer ID
        backend
        
        % Dimensions
        input_dim
        output_dim
        weight_dim
        
        % Activation
        act_name

    end
    
    methods
        %%
        function obj = unflatten(input_dim, output_dim)
            % UNFLATTEN the construction funciton for a fully connect layer
            % input_dim: INPUT, a scalar
            % output_dim: INPUT, 1x3 vector (no. of rows, cols, depths)
            
            if input_dim ~= prod(output_dim)
                error('Dimension mismatch');
            end
            
            obj.input_dim=input_dim;
            obj.output_dim=output_dim;
            
            obj.set_weight_dim;
            
        end
        %%
        function set_weight_dim(obj)
            % For func dense, set weights dimension
            % Num of rows = output dim
            % Num of col = input dim + 1 (if physical bias)
            
            obj.weight_dim = [0 0];
            
        end
        %%
        function initialize(obj)
        end
        %%
        function y_out = call(obj, x_in)
            % CALL The forward pass of the layer
            % x_in:   The input of the layer (2D)
            % y_out:  The output (4D)
            
            % Batch size
            n = size(x_in, 2);
            
            % Forward propogation                
            % (1) None-recurrent case (x_in 2D, size(x_in, 3)=1)
            % (2) Recurrent case (x_in 3D, size(x_in, 3)=time points)
            y_out = reshape(x_in, [obj.output_dim, n]);
            
        end
        %%
        function [grads, dx] = calc_gradients(obj, dy)
            % CALC_GRADIENTS: The backward pass of the layer
            %   dy:     The delta on this layer output (4D)
            %   grads:     (dW) The weight gradient in this layer
            %   dx:     The delta for previous layer (2D)
            
            grads = [];
            
            % Batch size
            n = size(dy, 4);
            
            % Back propogation (new deltas)
            dx = reshape(dy, [obj.input_dim, n]);
        end
    end
end