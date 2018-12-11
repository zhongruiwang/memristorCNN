classdef conv2D < layer
    % This class provides various activation functions
    
    properties
        nlayer % current layer number
        backend
        
        padding % Pad or not
        
        bias_config % a two dimensional array: [ratio rep]
        
        % dimensions
        input_dim % 3D
        output_dim % 3D
        kernel_size % 4D
        weight_dim % 2D flatten of kernel
        strides % 2D
        pad_size_pre % 2D
        pad_size_post % 2D
        
        % activation function names
        act_name
        
        % histories for back propogation
        x_in_padded_history
        y_out_history
        
    end
    
    methods
        function obj = conv2D( kernel_size, varargin )
            % CONV2D the construction funciton for a 2D convolution layer.
            %
            % kernel_size: 4D array (height*width*depth*population)
            % input_dim: 3D array (height*width*depth)
            % stride: 2D array (height stride, width stride)
            % padding: 'valid' or 'same'
            
            okargs = {'input_dim', 'strides', 'padding', 'activation', 'bias_config'};
            defaults = {NaN, [1 1], 'valid', 'relu', [1 1]};
            [obj.input_dim, obj.strides, obj.padding, obj.act_name, obj.bias_config] =...
                internal.stats.parseArgs(okargs, defaults, varargin{:});
                        
            obj.kernel_size = kernel_size;
            
            % If there is input_dim, auto SET weight_dim & output_dim
            if ~isnan(obj.input_dim)
                obj.set_weight_dim();
            end
            
            % History initialize
            obj.initialize();
        end
        %%
        function initialize(obj)
            % INITIALIZE initialize history (forward pass)
            
            obj.x_in_padded_history = [];
            obj.y_out_history = [];
        end
        %%
        function set_weight_dim(obj)
            % SET_DIM sets the output dimension, the padding, and the
            % flattend weight matrix dimension
            
            % Case with padding
            if strcmp(obj.padding,'same')
                
                % output_dim = [strides(y) strides(x) num_kernels]
                obj.output_dim = [ceil([obj.input_dim(1)/obj.strides(1),...
                    obj.input_dim(2)/obj.strides(2)]),...
                    obj.kernel_size(4)];
                
                % Padding size, pre/post size (2D)
                pad_size=([obj.output_dim(1), obj.output_dim(2)]-1).*obj.strides...
                    +[obj.kernel_size(1), obj.kernel_size(2)]...
                    -[obj.input_dim(1), obj.input_dim(2)];
                obj.pad_size_pre=floor(pad_size/2); % numb of pre-0s <= num of post-0s
                obj.pad_size_post=pad_size-obj.pad_size_pre;
                
            % Case without padding
            elseif strcmp(obj.padding, 'valid')
                
                obj.output_dim = [floor([(obj.input_dim(1)-obj.kernel_size(1))/obj.strides(1),...
                    (obj.input_dim(2)-obj.kernel_size(2))/obj.strides(2)]) + 1,...
                    obj.kernel_size(4)];
                
                % No padding
                obj.pad_size_pre=[0,0];
                obj.pad_size_post=[0,0];
                
            else
                error('Wrong padding scheme.');
            end
            
            % SET flatten weight (2D) dimension, based on kernel (4D)
            obj.weight_dim = [obj.kernel_size(4),...
                obj.kernel_size(1)*obj.kernel_size(2)*obj.kernel_size(3)+obj.bias_config(2)];
            
        end
        %%
        function y_out = call(obj, x_in)
            % CALL The forward pass of the layer
            % x_in: Inputs, size(x_in) = input_dim
            % y_out: Outputs, size(y_out) = output_dim
            
            % Input padding
            x_in_padded = x_in;
            if strcmp(obj.padding, 'same')
                x_in_padded = padarray(x_in_padded, obj.pad_size_pre, 'pre');
                x_in_padded = padarray(x_in_padded, obj.pad_size_post, 'post');
            end
            
            % HARDWARE call (conv2d)
            z = obj.backend.xcorr3d( x_in_padded, obj.bias_config,...
                obj.kernel_size, obj.output_dim, obj.strides, obj.nlayer );
            
            % Activations
            act = activations.get( obj.act_name, 'act');
            y_out = act( z );
            
            %  Store the output for future backpropogation
            % (note the DIM=5 is because y_out x_in_padded are 4D)
            
            obj.y_out_history = cat(5, obj.y_out_history, y_out);
            obj.x_in_padded_history = cat(5, obj.x_in_padded_history, x_in_padded);
            
        end
        %%
        function [grads, dx] = calc_gradients(obj, dy)
            % CALC_GRADIENTS The backward pass of the layer
            % dy: The delta on this layer (height*width*kernels*batch_size)
            % grads: The weight gradient in this layer (sum of the batch)
            % dx: The delta for previous layer
            
            % Pop out historical input/output
            [y_out, obj.y_out_history] = obj.history_pop(obj.y_out_history);
            [x_in_padded, obj.x_in_padded_history] = obj.history_pop(obj.x_in_padded_history);
            
            % Activation
            if ~contains(obj.act_name, 'softmax')
                
                % Fetch activation func derivative func
                act = activations.get(obj.act_name, 'deriv');
                
                % from deltas (after activation) to nodes before activation
                dy = dy .* act(y_out);
            end
            
            % grads (dE/dWeight), same dimension with kernel
            grads = zeros(obj.kernel_size);
            
            for row = 1:obj.kernel_size(1)
                for col = 1:obj.kernel_size(2)
                    
                    % Input lattice (x_in_padded) selection
                    input_row_select = row : obj.strides(1) : row+obj.strides(1)*(obj.output_dim(1)-1);
                    input_col_select = col : obj.strides(2) : col+obj.strides(2)*(obj.output_dim(2)-1);
                    
                    input_select = x_in_padded(input_row_select,input_col_select,:,:);
                    
                    for kernel_depth = 1:obj.kernel_size(3)
                        
                        % The kernelID (depth) of dE/dy
                        for kernel_ID = 1:obj.kernel_size(4)
                        
                            % Note the whole output (dy) is selected
                            % Summed over all samples of the input batch
                        
                            temp = dy(:,:,kernel_ID,:).*input_select(:,:,kernel_depth,:);
                            grads(row,col,kernel_depth,kernel_ID) = sum(temp(:));
                        end
                    end
                end
            end
            
            % bias errors (1D)
            db = sum( sum (sum (dy, 4), 2), 1);
            db = db(:)';
            
            % flatten grads (4D to 2D) + bias
            grads = reshape(grads, obj.kernel_size(1) * obj.kernel_size(2)...
                *obj.kernel_size(3), obj.kernel_size(4));
            grads = [grads; repmat(db, obj.bias_config(2), 1)];
            grads = grads';
            
            % Deltas backpropagation
            dx = obj.backend.xcorr3d_reverse( dy, obj.input_dim, ...
                obj.pad_size_pre, obj.kernel_size, obj.strides, obj.nlayer);
           
        end
    end
end