classdef conv2DLSTM < recurrent_layer
    % This class provides various activation functions
    properties
        nlayer      % current layer number
        backend
        
        bias_config % a two dimensional array: [ratio rep]
        
        % dimensions
        input_dim % 3D
        output_dim % 3D
        
        kernel_size_x % 4D
        kernel_size_h % 4D
        
        weight_dim % Combined
        
        strides_x % 2D, Note strides_h [1 1] constant
        
        padding_x % 'Same' or 'Valid'
        
        pad_size_pre_x % 2D
        pad_size_post_x % 2D
        
        pad_size_pre_h % 2D
        pad_size_post_h % 2D        
        
        % activation function names
        act_name
        recur_act_name
        
        % recurrent inputs
        y_pre
        hc_pre
        
        %
        dy_next
        dhc_next
        
        % histories for back propogation
        x_in_padded_history
        y_pre_padded_history
            
        ha_history
        hi_history
        hf_history
        ho_history
        
        hc_history
        hc_pre_history

    end
    
    methods
        %%
        function obj = conv2DLSTM( kernel_size_x, kernel_size_h, varargin )
            % conv2DLSTM the construction funciton for a LSTM layer.
            %
            % kernel_size: 4D array (height*width*depth*population)
            % input_dim: 3D array (height*width*depth)
            % stride: 2D array (height stride, width stride)
            % padding: 'valid' or 'same'
            
            okargs = {'input_dim', 'strides_x', 'padding_x',...
                'activation', 'recurrent_activation', 'bias_config'};
            defaults = {NaN, [1 1], 'valid',...
                'tanh', 'sigmoid', [1 1]};
            [obj.input_dim, obj.strides_x, obj.padding_x,...
                obj.act_name, obj.recur_act_name, obj.bias_config] =...
                internal.stats.parseArgs(okargs, defaults, varargin{:});
            
            % Kernels of input / recurrent input
            obj.kernel_size_x = kernel_size_x;
            if kernel_size_h(3) ~= kernel_size_h(4)
                error('Kernel h depth should = number');
            end
            obj.kernel_size_h = kernel_size_h;
            
            % If there is input_dim, auto SET weight_dim & output_dim
            if ~isnan(obj.input_dim)
                obj.set_weight_dim();
            end
            
            % History initialize
            obj.initialize();
            
        end
        %%
        function initialize(obj)
            obj.y_pre = [];
            obj.hc_pre = [];
            
            obj.dy_next = zeros( obj.output_dim );
            obj.dhc_next = zeros( obj.output_dim );
            
            obj.x_in_padded_history = [];
            obj.y_pre_padded_history = [];
            
            obj.ha_history = [];
            obj.hi_history = [];
            obj.hf_history = [];
            obj.ho_history = [];

            obj.hc_history = [];
            obj.hc_pre_history = [];
        end       
        %%
        function set_weight_dim(obj)
            % SET_DIM sets the output dimension, the padding, and the
            % flattend weight matrix dimension
            
            % Case with padding
            if strcmp(obj.padding_x,'same')
                
                % output_dim = [strides(y) strides(x) num_kernels]
                obj.output_dim = [ceil([obj.input_dim(1)/obj.strides_x(1),...
                    obj.input_dim(2)/obj.strides_x(2)]),...
                    obj.kernel_size_x(4)];
                
                % Padding size, pre/post size (2D)
                pad_size=([obj.output_dim(1), obj.output_dim(2)]-1).*obj.strides_x...
                    +[obj.kernel_size_x(1), obj.kernel_size_x(2)]...
                    -[obj.input_dim(1), obj.input_dim(2)];
                obj.pad_size_pre_x=floor(pad_size/2); % numb of pre-0s <= num of post-0s
                obj.pad_size_post_x=pad_size-obj.pad_size_pre_x;
                
            % Case without padding
            elseif strcmp(obj.padding_x, 'valid')
                
                obj.output_dim = [floor([(obj.input_dim(1)-obj.kernel_size_x(1))/obj.strides_x(1),...
                    (obj.input_dim(2)-obj.kernel_size_x(2))/obj.strides_x(2)]) + 1,...
                    obj.kernel_size_x(4)];
                
                % No padding
                obj.pad_size_pre_x=[0,0];
                obj.pad_size_post_x=[0,0];
                
            else
                error('Wrong padding scheme.');
            end
            
            % See if the output_dim(3) same with kernel_h depth
            if obj.output_dim(3) ~= obj.kernel_size_h(3)
                error('Wrong kernel depth');
            end
            
            % (h must padded) Padding size, pre/post size (2D)
            pad_size=[obj.kernel_size_h(1), obj.kernel_size_h(2)]-1;
            obj.pad_size_pre_h=floor(pad_size/2); % numb of pre-0s <= num of post-0s
            obj.pad_size_post_h=pad_size-obj.pad_size_pre_h;
            
            % SET flatten weight (2D) dimension
            obj.weight_dim = [4*obj.kernel_size_x(4), ...
                obj.kernel_size_x(1)*obj.kernel_size_x(2)*obj.kernel_size_x(3)+...
                obj.kernel_size_h(1)*obj.kernel_size_h(2)*obj.kernel_size_h(3)+...
                obj.bias_config(2)];
            
        end
        %%
        function y_out = call(obj, x_in)
            % CALL The forward pass of the layer
            % x_in: Inputs, size(x_in) = input_dim
            % y_out: Outputs, size(y_out) = output_dim
            
            % Time zero (add ht-1 part)
            n = size(x_in, 4); % batch size
            if isempty( obj.y_pre )
                obj.y_pre = zeros( [obj.output_dim, n] );
                obj.hc_pre = zeros( [obj.output_dim, n] );
            end
            
            % Input padding
            x_in_padded = x_in;
            if strcmp(obj.padding_x, 'same')
                x_in_padded = padarray(x_in_padded, obj.pad_size_pre_x, 'pre');
                x_in_padded = padarray(x_in_padded, obj.pad_size_post_x, 'post');
            end
            
            % Recurrent input padding (mandantory)
            y_pre_padded = obj.y_pre;
            y_pre_padded = padarray(y_pre_padded, obj.pad_size_pre_h, 'pre');
            y_pre_padded = padarray(y_pre_padded, obj.pad_size_post_h, 'post');
            
            % HARDWARE call (conv2d)
            z = obj.backend.xcorr3dLSTM( x_in_padded, y_pre_padded, obj.bias_config,...
                obj.kernel_size_x, obj.kernel_size_h, obj.output_dim,...
                obj.strides_x, obj.nlayer );
            
            % -------------------------------------------------------------
            % LSTM activations
            act = activations.get( obj.act_name, 'act');
            recur_act = activations.get( obj.recur_act_name, 'act');
            
            % (historical) activation, input / forget / output gates
            ha =       act( z(:, :, 1 : 1 * obj.output_dim(3), :) );
            hi = recur_act( z(:, :, 1 * obj.output_dim(3) + 1 : 2 * obj.output_dim(3), :) );
            hf = recur_act( z(:, :, 2 * obj.output_dim(3) + 1 : 3 * obj.output_dim(3), :) );
            ho = recur_act( z(:, :, 3 * obj.output_dim(3) + 1 : 4 * obj.output_dim(3), :) );
            
            % internal state c(t) and h(t)
            hc = hi .* ha + hf .* obj.hc_pre;
            y_out  = ho .* act(hc);
            
            % History x(t), h(t) a(t), i(t), f(t), o(t), tan(c(t)), c(t-1)
            obj.x_in_padded_history = cat(5, obj.x_in_padded_history, x_in_padded);
            obj.y_pre_padded_history = cat(5, obj.y_pre_padded_history, y_pre_padded);
            obj.ha_history = cat(5, obj.ha_history, ha); 
            obj.hi_history = cat(5, obj.hi_history, hi);
            obj.hf_history = cat(5, obj.hf_history, hf);
            obj.ho_history = cat(5, obj.ho_history, ho);
            
            obj.hc_history = cat(5, obj.hc_history, act(hc)); % tanh (c(t))
            obj.hc_pre_history = cat(5, obj.hc_pre_history, obj.hc_pre); % c(t-1)
            
            % Recurrent inputs
            obj.y_pre = y_out;
            obj.hc_pre = hc;
        end
        %%
        function [grads, dx] = calc_gradients(obj, dy)
            % CALC_GRADIENTS: The backward pass of the layer
            % Input:
            %   dy:     The delta on this layer
            % Output:
            %   dW:     The weight gradient in this layer
            %   dx:     The delta for previous layer
            
            % Get history: tanh(c(t)), c(t-1)
            [hc,  obj.hc_history] = obj.history_pop( obj.hc_history);
            [hc_pre_,  obj.hc_pre_history] = obj.history_pop( obj.hc_pre_history);
            
            % Get history: a(t), i(t), f(t), o(t)
            [ha,  obj.ha_history] = obj.history_pop( obj.ha_history);
            [hi,  obj.hi_history] = obj.history_pop( obj.hi_history);
            [hf,  obj.hf_history] = obj.history_pop( obj.hf_history);
            [ho,  obj.ho_history] = obj.history_pop( obj.ho_history);
            
            % Get history : Inpnt / Recurrent input (x(t), h(t-1)
            [x_in_padded,  obj.x_in_padded_history] = obj.history_pop( obj.x_in_padded_history);
            [y_pre_padded,  obj.y_pre_padded_history] = obj.history_pop( obj.y_pre_padded_history);
            
            % LSTM activations
            act = activations.get( obj.act_name, 'deriv');
            recur_act = activations.get( obj.recur_act_name, 'deriv');
            
            % dE/dh(t), contributed by propagation and time backpropagation
            dy = dy + obj.dy_next;
            
            % dE/dO(t), dE/dC(t)
            dho = dy  .* hc .* recur_act( ho ); % hc = tanh(C(t)), recur_act = sigma'
            dhc = dy  .* ho .* act(hc) + obj.dhc_next; % ho = O(t), act(hc) = tanh'(tanh(C(t))= 1-tanh^2(C(t))           
            dha = dhc .* hi .* act(ha); % hi = i(t), act(ha) = tanh'(a(t))
            dhi = dhc .* ha .* recur_act(hi); % ha = a(t), recur_act(hi) = sigma'(i(t))
            dhf = dhc .* hc_pre_ .* recur_act(hf); % hc_pre = C(t-1), recur_act(hf) = sigma'(f(t))
            
            % Get dE/dC(t-1)
            obj.dhc_next = dhc .* hf; % hf = f(t), 
            
            % You get the deltas right after convolution blocks
            dz = cat(3, dha, dhi, dhf, dho); % Combine along DIM =3 (make number of kernels *4)
            
            % grads (dE/dWeight), same dimension with kernel
            grads_x = zeros(obj.kernel_size_x(1), obj.kernel_size_x(2), ...
                obj.kernel_size_x(3),obj.kernel_size_x(4)*4); % last dimension *4
            
            % grads (dE/dWeight), same dimension with kernel
            grads_h = zeros(obj.kernel_size_h(1), obj.kernel_size_h(2), ...
                obj.kernel_size_h(3),obj.kernel_size_h(4)*4); % last dimension *4
            
            % Backpropgate through input convolution
            for row = 1:obj.kernel_size_x(1)
                for col = 1:obj.kernel_size_x(2)
                    
                    % Input lattice (x_in_padded) selection ___ for x
                    input_row_select = row:obj.strides_x(1):row+obj.strides_x(1)*obj.output_dim(1)-1;
                    input_col_select = col:obj.strides_x(2):col+obj.strides_x(2)*obj.output_dim(2)-1;                   
                    input_select_x = x_in_padded(input_row_select,input_col_select,:,:);
                    
                    for kernel_depth_x = 1:obj.kernel_size_x(3)
                        
                        % The kernelID (depth) of dE/dy
                        for kernel_ID = 1:obj.kernel_size_x(4)*4
                        
                            temp = dz(:,:,kernel_ID,:).*input_select_x(:,:,kernel_depth_x,:);
                            grads_x(row,col,kernel_depth_x,kernel_ID) = sum(sum(sum(temp)));
                        end 
                    end    
                end
            end
            
            % Backpropgate through recurrent input convolution
            for row = 1:obj.kernel_size_h(1)
                for col = 1:obj.kernel_size_h(2)
                    
                    % Input lattice (x_in_padded) selection ___ for h(t-1)
                    input_row_select = row:row+obj.output_dim(1)-1;
                    input_col_select = col:col+obj.output_dim(2)-1;            
                    input_select_h = y_pre_padded(input_row_select,input_col_select,:,:);
                
                    for kernel_depth_h = 1:obj.kernel_size_h(3)
                        
                        % The kernelID (depth) of dE/dy
                        for kernel_ID = 1:obj.kernel_size_h(4)*4
                            
                            temp = dz(:,:,kernel_ID,:).*input_select_h(:,:,kernel_depth_h,:);
                            grads_h(row,col,kernel_depth_h,kernel_ID) = sum(sum(sum(temp)));
                        end           
                    end    
                end
            end
            
            % bias errors (1D)
            db = sum( sum (sum (dz, 4), 2), 1);
            db = db(:)';
            
            % Combine flatten grads_x and grads_h (4D to 2D) + bias
            grads = [reshape(grads_x, obj.kernel_size_x(1) * obj.kernel_size_x(2)...
                *obj.kernel_size_x(3), obj.kernel_size_x(4)*4);...
                reshape(grads_h, obj.kernel_size_h(1) * obj.kernel_size_h(2)...
                *obj.kernel_size_h(3), obj.kernel_size_h(4)*4)];
            grads = [grads; repmat(db, obj.bias_config(2), 1)];
            grads = grads';
            
            % delta inputs          
            [dx, obj.dy_next] = obj.backend.xcorr3dLSTM_reverse(dz,...
                obj.input_dim, obj.output_dim, obj.pad_size_pre_x, obj.pad_size_pre_h,...
                obj.kernel_size_x, obj.kernel_size_h, obj.strides_x, obj.nlayer);
            
        end
    end
end