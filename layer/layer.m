classdef (Abstract) layer < handle
    % An abstract class for neural network layers
    
    properties (Abstract)
        nlayer      % current maximum layer number in the backend
        backend      % Backend interface
        
        % Dimensions
        input_dim
        output_dim
        weight_dim
        
        % Activation
        act_name
    end
    
    methods (Abstract)
        
        % Foward pass (input: x_in)
        [y_out, probes] = call(obj, x_in);
        
        % Error gradient (input: deltas, ?)
        [dW, dx] = calc_gradients(obj, dy, probes);
        
        % Initialize layer?
        initialize(obj);
    end
    
    methods (Access = protected )
        
        % Pop out a historical weight (history ID), only accessible to
        % local methods and subclass
        function [item, history] = history_pop(~, history)
        
            % Check history dimension (either 5, or 3)...
            if isempty(history)
                error('Nothing in the layer history!');
            elseif and(size(history, 4)==1, size(history, 5)==1)
                
                % Pop out one last history 2D
                item = history(:,:,end);
                history(:,:,end) = [];
                
            else
                
                % Pop out one last history 4D
                item = history(:,:,:,:,end);
                history(:,:,:,:,end) = [];
                
            end            
        end
    end
end