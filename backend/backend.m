% A abstract class for various backend accelerators

classdef (Abstract) backend < handle

    methods (Abstract)
        
        % Initialized the weights of the network
        initialize_weights(obj, varargin)
        
        % Change weight (given only delta weight cell arrays)
        update(obj, dWs);
        
        % Add a layer (given weight dimension vector)
        add_layer(obj, weight_dim);
        
        % Forward pass for dense/LSTM (vectors, and the layer ID)
        output = multiply(obj, vec, layer);
        
        % Convolution forward pass
        output = xcorr3d(obj, input, bias_config, kernel_size, output_dim,...
                strides, layer)
        
        % Backward pass for dense/LSTM (vector, and the layer ID)
        output = multiply_reverse(obj, vec, layer);
        
        % Convolution backward pass
        output = xcorr3d_reverse(obj, dy, input_dim, pad_size_pre,...
                kernel_size, strides, layer)
         
        % Check whether layer ID in proper range
        check_layer(obj, layer );
    end
end