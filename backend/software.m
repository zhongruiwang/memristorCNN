classdef software < backend
    % Pure MATLAB backend
    
    properties
        %store the weights (cell vector)
        W
    end
    
    methods
        
        %%
        function obj = software( )
            % Define object
            
            obj.W = {};
        end
        %%
        function add_layer(obj, weight_dim, varargin)
            % ADD_LAYER add another layer to the software backend.
            
            % Add one layer (vertical differential pair)
            obj.W{end+1} = (2 * rand( weight_dim ) - 1);
            
            % Normalization (why sqrt(num of columns)?)
            obj.W{end} = obj.W{end} / sqrt( size(obj.W{end}, 2) );
        end
        %%
        function initialize_weights(~, varargin)
        end
        %%
        function update(obj, dW)
            %UPDATE update weights by layers
            
            for l = 1:numel( dW )
%               obj.W{l} = obj.W{l} + dW{l};
                
                % Simulating noise writing
                noise = 0.05*(2*rand(size(dW{l}))-1);
                obj.W{l} = (obj.W{l} + dW{l}).*(1+noise);
                
            end
        end
        %%
        function check_layer(obj, layer )
            % Check if the layer ID is within legal range
            
            if layer > numel(obj.W)
                error(['layer number should be less than ' num2str(numel(obj.W))]);
            end
        end
        %%
        function output = multiply(obj, vec, layer)
            % MULTIPLY: forward pass of fully connected / LSTM layers
            % vec (INPUT): 2D matrix, each column per sample
            % layer (INPUT): layer number
            
            % Check
            obj.check_layer(layer);
            % Vector-matrix multiplication
            output = obj.W{layer} * vec;
        end
        %%
        function output = xcorr3d(obj, input, bias_config, kernel_size, output_dim,...
                strides, layer)
            % XCORR3D convolution forward pass
            % input: Input 4D (height*width*depth*batch_size)
            % bias_config: bias_config (ratio, repeats)
            % kernel_size: 4D (height*width*depth*population)
            % output_dim: 4D (height*width*population*batch_size)
            
            % Initialize output
            output = zeros([output_dim, size(input,4)]);
            
            % Convolution 2d
            for sample_ID = 1:size(input,4)
                for row = 1:output_dim(1)
                    for col = 1:output_dim(2)
                        
                        % Input row/column selections (1D)
                        row_select = 1+strides(1)*(row-1):strides(1)*(row-1)+kernel_size(1);
                        col_select = 1+strides(2)*(col-1):strides(2)*(col-1)+kernel_size(2);
                        temp = input(row_select,col_select,:,sample_ID);
                        
                        % Convert input from 3D to 1D
                        temp = temp(:);
                        
                        % Add bias
                        temp_b = [temp; repmat(bias_config(1), bias_config(2), 1)];
                        
                        % Physical product
                        output(row,col,:,sample_ID) = obj.W{layer} * temp_b;
                        
                    end
                end
            end
        end
        %%
        function output = xcorr3dLSTM(obj, input_x, input_h, bias_config,...
                kernel_size_x, kernel_size_h, output_dim, strides_x, layer)
            % XCORR3D convolution forward pass
            % input: Input 4D (height*width*depth*batch_size)
            % bias_config: bias_config (ratio, repeats)
            % kernel_size: 4D (height*width*depth*population)
            % output_dim: 4D (height*width*population*batch_size)
            
            % Initialize output (note size(input_x,4)=size(input_h,4)!)
            output = zeros([output_dim(1), output_dim(2),...
                output_dim(3)*4, size(input_x,4)]);
            
            % Convolution 2d
            for sample_ID = 1:size(input_x,4)
                for row = 1:output_dim(1)
                    for col = 1:output_dim(2)
                        
                        % Input row/column selections (1D) ___ for x
                        row_select = 1+strides_x(1)*(row-1):strides_x(1)*(row-1)+kernel_size_x(1);
                        col_select = 1+strides_x(2)*(col-1):strides_x(2)*(col-1)+kernel_size_x(2);
                        temp_x = input_x(row_select,col_select,:,sample_ID);
                        
                        % Input row/column selections (1D) ___ for h
                        row_select = 1+(row-1):(row-1)+kernel_size_h(1);
                        col_select = 1+(col-1):(col-1)+kernel_size_h(2);
                        temp_h = input_h(row_select,col_select,:,sample_ID);
                        
                        % Convert input from 3D to 1D
                        temp = [temp_x(:); temp_h(:)];
                        
                        % Add bias
                        temp_b = [temp; repmat(bias_config(1), bias_config(2), 1)];
                        
                        % Physical product
                        output(row,col,:,sample_ID) = obj.W{layer} * temp_b;
                        
                    end
                end
            end
        end
        %%
        function output = multiply_reverse(obj, vec, layer)
            % Backward pass
            
            % Check
            obj.check_layer(layer);
            % Vector-matrix multiplication
            output = obj.W{layer}.' * vec;
        end
        %%
        function output = xcorr3d_reverse(obj, dy, input_dim, pad_size_pre,...
                kernel_size, strides, layer)
            % XCORR3D_REVERSE delta backpropagate in CNN layer
            
            % Initialize output (dE/dx of the front layer)
            output = zeros([input_dim size(dy,4)]);
            
            % 4D kernel based on 2D weight matrix
            temp=obj.W{layer}';
            temp=temp(1:kernel_size(1)*kernel_size(2)*kernel_size(3),:);
            kernel=reshape(temp,kernel_size);
            
            % Visit all dx (row/col refer to padded array)
            for row = pad_size_pre(1)+1 : input_dim(1)+pad_size_pre(1)
                for col = pad_size_pre(2)+1 : input_dim(2)+pad_size_pre(2)
                    
                    % Select the output lattice where the input(px,py) contributed
                    dy_row_select = ceil((row-kernel_size(1))/strides(1)+1):floor((row-1)/strides(1)+1);
                    dy_row_select = dy_row_select(dy_row_select>0 & dy_row_select<=size(dy,1)); % Enforce range
                    
                    dy_col_select = ceil((col-kernel_size(2))/strides(2)+1):floor((col-1)/strides(2)+1);
                    dy_col_select = dy_col_select(dy_col_select>0 & dy_col_select<=size(dy,2)); % Enforce range
                    
                    % Select the corresponding weight
                    kernel_row_select = row-strides(1)*(dy_row_select-1);
                    kernel_col_select = col-strides(2)*(dy_col_select-1);
                    
                    for depth=1:kernel_size(3)
                        for sampleID = 1:size(dy,4)
                            
                            % Dim: rows * cols * depts * 1
                            dy_select = dy(dy_row_select,dy_col_select,:,sampleID);
                            % Dim: rows * cols * 1 * kernel_numbers
                            kernel_select=kernel(kernel_row_select, kernel_col_select,depth,:);
                            
                            temp = dy_select.*kernel_select;
                            
                            output(row-pad_size_pre(1),col-pad_size_pre(2),depth,sampleID)= sum(temp(:));
                        end
                    end
                end
            end
        end
        %%
        function [output_x, output_h] = xcorr3dLSTM_reverse(obj, dy,...
                input_dim_x, input_dim_h, pad_size_pre_x, pad_size_pre_h,...
                kernel_size_x, kernel_size_h, strides_x, layer)
            % XCORR3DLSTM_REVERSE delta backpropagate in conv2DLSTM layer
            
            % Initialize output (dE/dx of the front layer)
            output_x = zeros([input_dim_x, size(dy,4)]);
            output_h = zeros([input_dim_h, size(dy,4)]);
            
            % 4D kernel_x and kernel_h based on 2D weight matrix
            temp=obj.W{layer}';
            temp_index_x=prod(kernel_size_x(1:3));
            temp_index_h=prod(kernel_size_h(1:3));
            temp_x=temp(1:temp_index_x,:);
            kernel_x=reshape(temp_x,kernel_size_x(1), kernel_size_x(2),...
                kernel_size_x(3), kernel_size_x(4)*4);
            temp_h=temp(temp_index_x+1:temp_index_x+temp_index_h,:);
            kernel_h=reshape(temp_h,kernel_size_h(1), kernel_size_h(2),...
                kernel_size_h(3), kernel_size_h(4)*4);
            
            % Visit all dx (row/col refer to padded array) ___ input x(t)
            for row = pad_size_pre_x(1)+1 : input_dim_x(1)+pad_size_pre_x(1)
                for col = pad_size_pre_x(2)+1 : input_dim_x(2)+pad_size_pre_x(2)
                    
                    % Select the output lattice where the input(px,py) contributed
                    dy_row_select = ceil((row-kernel_size_x(1))/strides_x(1)+1):floor((row-1)/strides_x(1)+1);
                    dy_row_select = dy_row_select(dy_row_select>0 & dy_row_select<=size(dy,1)); % Enforce range
                    
                    dy_col_select = ceil((col-kernel_size_x(2))/strides_x(2)+1):floor((col-1)/strides_x(2)+1);
                    dy_col_select = dy_col_select(dy_col_select>0 & dy_col_select<=size(dy,2)); % Enforce range
                    
                    % Select the corresponding weight
                    kernel_row_select = row-strides_x(1)*(dy_row_select-1);
                    kernel_col_select = col-strides_x(2)*(dy_col_select-1);
                    
                    for depth=1:kernel_size_x(3)
                        for sampleID = 1:size(dy,4)
                            
                            % Dim: rows * cols * depts * 1
                            dy_select = dy(dy_row_select,dy_col_select,:,sampleID);
                            % Dim: rows * cols * 1 * kernel_numbers
                            kernel_select_x=kernel_x(kernel_row_select, kernel_col_select,depth,:);
                            
                            temp = dy_select.*kernel_select_x;
                            
                            output_x(row-pad_size_pre_x(1),col-pad_size_pre_x(2),...
                                depth,sampleID)= sum(temp(:));
                        end
                    end
                end
            end
            
            % Visit all dx (row/col refer to padded array) _ recurrent input h(t-1)
            for row = pad_size_pre_h(1)+1 : input_dim_h(1)+pad_size_pre_h(1)
                for col = pad_size_pre_h(2)+1 : input_dim_h(2)+pad_size_pre_h(2)
                    
                    % Select the output lattice where the input(px,py) contributed
                    dy_row_select = ceil((row-kernel_size_h(1))+1):floor((row-1)+1);
                    dy_row_select = dy_row_select(dy_row_select>0 & dy_row_select<=size(dy,1)); % Enforce range
                    
                    dy_col_select = ceil((col-kernel_size_h(2))+1):floor((col-1)+1);
                    dy_col_select = dy_col_select(dy_col_select>0 & dy_col_select<=size(dy,2)); % Enforce range
                    
                    % Select the corresponding weight
                    kernel_row_select = row-(dy_row_select-1);
                    kernel_col_select = col-(dy_col_select-1);
                    
                    for depth=1:kernel_size_h(3)
                        for sampleID = 1:size(dy,4)
                            
                            % Dim: rows * cols * depts * 1
                            dy_select = dy(dy_row_select,dy_col_select,:,sampleID);
                            % Dim: rows * cols * 1 * kernel_numbers
                            kernel_select_h=kernel_h(kernel_row_select, kernel_col_select,depth,:);
                            
                            temp = dy_select.*kernel_select_h;
                            
                            output_h(row-pad_size_pre_h(1),col-pad_size_pre_h(2),...
                                depth,sampleID)= sum(temp(:));
                        end
                    end
                end
            end
        end        
    end
end