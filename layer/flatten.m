classdef flatten < layer
    % Flatten layer (4D to 2D only)
    
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
        function obj = flatten()
            % FLATTEN the construction funciton for a fully connect layer
            
        end
        %%
        function set_weight_dim(obj)
            % SET_WEIGHT_DIM here just update output_dimension
            
            obj.weight_dim = [0 0];
            
            % (Update the output_dim here
            % output_dim (scalar), input_dim (1x3 vector)
            obj.output_dim = obj.input_dim(1)*obj.input_dim(2)...
                *obj.input_dim(3);
        end
        %%
        function initialize(obj)
        end
        %%
        function y_out = call(obj, x_in)
            % CALL The forward pass of the layer
            % x_in:   The input of the layer (4D)
            % y_out:  The output (2D flattend)
            
            % Batch size
            n = size(x_in, 4);
            
            % Forward propogation
            y_out = reshape(x_in, [obj.output_dim, n]);
            
        end
        %%
        function [grads, dx] = calc_gradients(obj, dy)
            % CALC_GRADIENTS: The backward pass of the layer
            %   dy:     The delta on this layer output (2D)
            %   grads:     (dW) The weight gradient in this layer
            %   dx:     The delta for previous layer (4D)
            
            grads = [];
            
            % Batch size
            n = size(dy, 2);
            
            % Back propogation (new deltas)
            dx = reshape(dy, [obj.input_dim, n]);
        end
    end
end