classdef upsampling2D < layer
    % Flatten layer (4D to 2D only)
    
    properties
        
        nlayer % current layer ID
        backend
        
        % Dimensions
        input_dim
        output_dim
        weight_dim
        repeat_size % (legacy)
        
        % Interpolation meshes and method
        inputmesh_row
        inputmesh_col
        outputmesh_row
        outputmesh_col
        interpolation

        % Activation (legacy)
        act_name

    end
    
    methods
        %%
        function obj = upsampling2D(repeat_size, varargin)
            % FLATTEN the construction funciton for a fully connect layer
            
            okargs = {'interpolation'};
            defaults = {'linear'}; % Default stride = pool_size
            [obj.interpolation] = internal.stats.parseArgs(okargs, defaults, varargin{:});
            
            % Validate repeat size
            if numel(repeat_size) == 2
                obj.repeat_size = repeat_size;
            else
                error('upsampling repeat_size example [2,2]');
            end
            
        end
        %%
        function initialize(obj)
            % INITIALIZE

        end
        %%
        function set_weight_dim(obj)
            % SET_DIM sets the output dimension, and the flattend weight
            % matrix dimension (legacy)
            
            obj.output_dim = obj.input_dim.*[obj.repeat_size 1];
            
            % Mesh
            [obj.inputmesh_row, obj.inputmesh_col] = meshgrid(...
                1:obj.input_dim(2), 1:obj.input_dim(1));
            [obj.outputmesh_row, obj.outputmesh_col] = meshgrid(...
                linspace(1, obj.input_dim(2), obj.output_dim(2)),...
                linspace(1, obj.input_dim(1), obj.output_dim(1)));
            
            % No weight
            obj.weight_dim = [0 0];
        
        end
        %%
        function y_out = call(obj, x_in)
            % CALL The forward pass of the layer
            % x_in:   The input of the layer (4D)
            % y_out:  The output (2D flattend)
            
            % Output initialization
            y_out=NaN([obj.output_dim, size(x_in,4)]);
            
            % 2D Interpolation
            for  sample_ID = 1:size(x_in, 4) 
                for depth = 1:obj.output_dim(3)
                
                     y_out(:,:,depth, sample_ID) = interp2(...
                         obj.inputmesh_row,  obj.inputmesh_col,...
                         x_in(:,:,depth, sample_ID),...
                         obj.outputmesh_row, obj.outputmesh_col,...
                         obj.interpolation);
                     
                end
            end
            
        end
        %%
        function [grads, dx] = calc_gradients(obj, dy)
            % CALC_GRADIENTS: The backward pass of the layer
            %   dy:     The delta on this layer
            %   grads:     (dW) The weight gradient in this layer
            %   dx:     The delta for previous layer
            
            % No gradient
            grads = [];
            
            % Initialize dx (dim: [input_dim + batch size])
            dx = zeros([obj.input_dim, size(dy, 4)]);
            
            % 2D Inverse Interpolation
            for  sample_ID = 1:size(dy, 4) 
                for depth = 1:obj.output_dim(3)
                
                     dx(:,:,depth, sample_ID) = interp2(...
                         obj.outputmesh_row,  obj.outputmesh_col,...
                         dy(:,:,depth, sample_ID),...
                         obj.inputmesh_row, obj.inputmesh_col,...
                         obj.interpolation);
                end
            end
            
        end
    end
end