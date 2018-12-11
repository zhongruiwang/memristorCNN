classdef maxpooling2D < layer
    % Flatten layer (4D to 2D only)
    
    properties
        
        nlayer % current layer ID
        backend
        
        % Padding
        padding
        
        % Dimensions
        input_dim
        output_dim
        weight_dim
        pool_size
        strides
        
        pad_size_pre
        pad_size_post
                
        % Activation
        act_name
        
        % history of winners
        x_in_win_history


    end
    
    methods
        %%
        function obj = maxpooling2D(pool_size, varargin)
            % FLATTEN the construction funciton for a fully connect layer
            
            okargs = {'strides', 'padding'};
            defaults = {pool_size, 'valid'}; % Default stride = pool_size
            [obj.strides, obj.padding] = internal.stats.parseArgs(okargs, defaults, varargin{:});
            
            obj.pool_size = pool_size;
            
        end
%%
        function initialize(obj)
            % INITIALIZE winner history
            
            obj.x_in_win_history = [];
        end
        %%
        function set_weight_dim(obj)
            % SET_DIM sets the output dimension, the padding, and the
            % flattend weight matrix dimension
            
            % Case with padding
            if strcmp(obj.padding,'same')
                
                % Size of the output (3D) =
                % strides(y-axis)*strides(x-axis)*depth
                obj.output_dim = [ceil([obj.input_dim(1)/obj.strides(1),...
                    obj.input_dim(2)/obj.strides(2)]),...
                    obj.input_dim(3)];
                
                % Padding size, pre/post size (2D)
                pad_size=([obj.output_dim(1), obj.output_dim(2)]-1).*obj.strides...
                    +[obj.pool_size(1), obj.pool_size(2)]...
                    -[obj.input_dim(1), obj.input_dim(2)];
                obj.pad_size_pre=floor(pad_size/2); % numb of pre-0s <= num of post-0s
                obj.pad_size_post=pad_size-obj.pad_size_pre;
                
            % Case without padding
            elseif strcmp(obj.padding, 'valid')
                
                obj.output_dim = [ceil([(obj.input_dim(1)-obj.pool_size(1)+1)/obj.strides(1),...
                    (obj.input_dim(2)-obj.pool_size(2)+1)/obj.strides(2)]),...
                    obj.input_dim(3)];
                
                % No padding
                obj.pad_size_pre=[0,0];
                obj.pad_size_post=[0,0];
                
            else
                error('Wrong padding scheme.');
            end
            
            % No weight
            obj.weight_dim = [0 0];
            
        end
        %%
        function y_out = call(obj, x_in)
            % CALL The forward pass of the layer
            % x_in:   The input of the layer (4D)
            % y_out:  The output (2D flattend)
            
            % Input padding (with -inf)
            x_in_padded = x_in;
            if strcmp(obj.padding, 'same')
                x_in_padded = padarray(x_in_padded, obj.pad_size_pre, -inf, 'pre');
                x_in_padded = padarray(x_in_padded, obj.pad_size_post, -inf, 'post');
            end
            
            % Output initialization
            y_out=zeros([obj.output_dim, size(x_in,4)]);
            
            % x_in_winner initializatoin
            x_in_win=cell([obj.output_dim, size(x_in,4)]);
            
            % Convolution 2d
            for row = 1:obj.output_dim(1)
                for col = 1:obj.output_dim(2)
            
                    % Input row/column selections (1D)
                    row_select = 1+obj.strides(1)*(row-1):obj.strides(1)*(row-1)+obj.pool_size(1);
                    col_select = 1+obj.strides(2)*(col-1):obj.strides(2)*(col-1)+obj.pool_size(2);
                    
                    for depth = 1:obj.output_dim(3)
                        for sample_ID = 1:size(x_in,4)
                            
                            % Picked area
                            temp = x_in_padded(row_select,col_select,depth,sample_ID);
                        
                            % Convert input from 3D to 1D
                            temp_value = max(temp(:));
                            
                            % Get max element coordinate
                            [temp_row, temp_col] = find(temp == temp_value);
                            
                            % Pooled output
                            y_out(row,col,depth,sample_ID) = temp_value;
                        
                            % Winner locations (no padding coordinate)
                            % Note that there many be more than 1 winners
                            row_win_no_pad = row_select(temp_row) - obj.pad_size_pre(1);
                            col_win_no_pad = col_select(temp_col) - obj.pad_size_pre(2);
                            
                            x_in_win{row,col,depth,sample_ID} = ...
                                [row_win_no_pad', col_win_no_pad'];
                        end
                    end
                end
            end
            % Store the output for future backpropogation
            % (note the DIM=3 is because x_in_padded are 4D)
            
            obj.x_in_win_history = cat(5, obj.x_in_win_history, x_in_win);
            
        end
        %%
        function [grads, dx] = calc_gradients(obj, dy)
            % CALC_GRADIENTS: The backward pass of the layer
            %   dy:     The delta on this layer
            %   grads:     (dW) The weight gradient in this layer
            %   dx:     The delta for previous layer
            
            % Pop out historical input/output
            [x_in_win, obj.x_in_win_history] = obj.history_pop(obj.x_in_win_history);
            
            % No gradient
            grads = [];
            
            % Initialize dx (dim: [input_dim + batch size])
            dx = zeros([obj.input_dim, size(dy, 4)]);
            
            % Back propogation (new deltas)
            for row = 1:obj.output_dim(1)
                for col = 1:obj.output_dim(2)
                    for depth = 1:obj.output_dim(3)
                        for sample_ID = 1:size(dy,4)
                            
                            % The winner location (two cols list)
                            temp = x_in_win{row, col, depth, sample_ID};
                            
                            % Assign the dx to winner(s)
                            dx(temp(:,1), temp(:,2), depth, sample_ID) = dy(row, col, depth, sample_ID);
                        end
                    end
                end
            end
            
        end
    end
end