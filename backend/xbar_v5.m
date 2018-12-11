classdef xbar_v5 < backend
    % This class maps the numerical values to physical ones to interface with
    % implementation of sim_array

    properties
        base        % multi_array object
        dp_rep      % Differential pair repetations (row / column-wise)
        dp_exceed   % true when verticle array size exceeds array bounds
        
        phys_layer_num  % no. of the physical layer (otherwise 0)
        
        % Blind update parameters
        th_set = 0;
        th_reset = -0;  % Should be negative
        
        Vg0 = 1.15;	    % Initial SET gate voltage
        Vg_max = 1.9;   % Max SET gate voltage
        Vg_min = 0.4;

        V_set = 2.5;    % Fixed SET voltage
        V_reset = 1.7;  % Fixed RESET voltage
        V_gate_reset = 5; % Fixed RESET gate voltage
        
        % Read parameters
        V_read = 0.2;
        
        % Mapping (Conductance to weight, Gate voltage to conductance)
        ratio_G_W = 100e-6;     % Conductance / Weight
        ratio_Vg_G = 1/100e-6;  % Delta_V_gate / Delta_conductancd
        
        % The array conductance for multiply reverse
        % update the value after weight update
        array_G
        
        % If true then plot
        draw = 0;
        
        % History
        V_gate_last; % tuned parameter determines the final G
        
        % History; save history if true
        save = 0;
        G_history = {};
        V_gate_history = {};
        V_reset_history = {};
        I_history = {};
        V_vec_history = {};
        
    end
    methods
        %%
        function obj = xbar_v5( base )
            obj.base = base;
        end
        
        %%
        function add_layer(obj, weight_dim, net_corner, layer_original, dp_rep)
            % ADD_LAYER add another layer to the software backend.
            
            % Add a physical layer for dense/lstm/conv2d/conv2dlstm
            if ~any(weight_dim(:))
                obj.phys_layer_num(layer_original)  = 0;
                return
            else
                % In case this layer is physical layer, the physical layer
                layer = numel( obj.base.subs )+1;
                obj.phys_layer_num(layer_original) = layer;
                
                % Differential pair (vertical) repeats                
                phys_size = fliplr(weight_dim) .* [2, 1] .* dp_rep;
                obj.dp_rep{layer} = dp_rep;
                
                % Is this layer need to be chopped (valid to FC only)
                temp = phys_size(1)/obj.base.net_size(1);
                if and(temp>1, temp<=2) % 0.5*tot_rows
                    obj.dp_exceed(layer) = 1;
                    phys_size = phys_size.*[0.5, 2];
                elseif temp<=1
                    obj.dp_exceed(layer) = 0;
                else
                    error('Do not support this DP configuration yet');
                end
                
                % Physically allocate the layer
                obj.base.add_sub(net_corner, phys_size);
            end
        end
        
        %%
        function initialize_weights(obj, varargin)
            okargs = {'draw', 'save'};
            defaults = {0, 0};
            [obj.draw, obj.save] = internal.stats.parseArgs(okargs, defaults, varargin{:});
            
            % SET and Save the initial SET voltages
            obj.V_gate_last = cellfun(@(x)zeros(x.net_size)+obj.Vg0, obj.base.subs,'UniformOutput',false);
            
            % update the initial conductance
            obj.base.update_subs('GND', obj.V_reset, obj.V_gate_reset); % RESET pulse
            obj.base.update_subs(obj.V_set, 'GND', obj.V_gate_last ); % SET pulse
            
            % Read and store the weight
            obj.read_conductance('mode', 'fast');
        end
        
        %%
        function update( obj, dWs_original )
            % UPDATE: update weight matrix
            
            % Remove dW (0x0 dimension) of "virtual" layers
            dWs = dWs_original(obj.phys_layer_num~=0);
            
            % Check number of dWs is same with number of physical layers
            nlayer = numel(dWs);
            
            if nlayer ~= numel(obj.base.subs)
                error('Wrong number of weight gradients');
            end
            
            % Initialize variables
            Vr = cell(1, nlayer);
            Vg = cell(1, nlayer);
            Vg_apply = Vg; % This is really applied voltage (not recorded)
            
            for layer = 1: numel(dWs)
                
                % Transpose
                dW = dWs{layer}';
                
                % Gradient scaling and to voltage conversion
                dV_temp = obj.ratio_Vg_G * obj.ratio_G_W * dW;
                
                % Vertical differential pair
                dV = NaN(size(dV_temp).*[2 1]);
                dV(1:2:end-1,:) = dV_temp; dV(2:2:end,:) = -dV_temp;                
                
                % Repeats of differential pairs
                dV = repmat(dV, obj.dp_rep{layer});
                
                % (FC only) If the layer is chopped 
                if obj.dp_exceed(layer) == 1
                    dV = [dV(1:end/2,:), dV(end/2+1:end,:)];
                end

                % RESET if dV is negative (or <th_reset)
                Vr{layer} = obj.V_reset .* (dV < obj.th_reset);
                
                % SET if (1) dV is postive (or > th_set) (2) Just RESET
                Vg{layer} = obj.V_gate_last{layer} + dV.*(dV>= obj.th_set | dV<=obj.th_reset);
                
                % Regulate the min and max SET gate voltage
                Vg{layer}(Vg{layer} > obj.Vg_max) = obj.Vg_max;
                Vg{layer}(Vg{layer} < obj.Vg_min) = obj.Vg_min;
                
                % Gate voltages applied (skip those do not need changes)
                Vg_apply{layer} = Vg{layer} .* (dV>obj.th_set | dV<obj.th_reset);

            end
            
            % Update (p1, p2 are pulse history)
            p1 = obj.base.update_subs('GND', Vr, obj.V_gate_reset); % RESET pulse
            p2 = obj.base.update_subs(obj.V_set, 'GND', Vg_apply); % SET pulse
                       
            % Save updated gate voltages
            obj.V_gate_last = Vg;
            
            % update the conductance for software backpropogation
            obj.read_conductance('mode', 'fast');
            
            % Plot
            if obj.draw
                figure(2); subplot(1,3,2);
                imagesc(p1{2}, [0 obj.V_reset]);colorbar;
                title('Reset voltages');
                
                subplot(1,3,3);
                imagesc(p2{3}, [0 obj.Vg_max]);colorbar;
                title('Gate voltage');
                drawnow;
            end
            
            % Save pulse history if needed
            if obj.save
                obj.V_gate_history{end+1} = p2{3};
                obj.V_reset_history{end+1} = p1{2};
            end
            
        end
        %%
        function fullG = read_conductance(obj, varargin)
            % READ_CONDUCTANCE read the conductances of all sub_arrays
            
            % Conductance read
            [obj.array_G , fullG] = obj.base.read_subs(varargin{:});
            
            if obj.draw
                figure(2); subplot(1,3,1);
                imagesc(fullG); colorbar;
                title('Conductance');
                drawnow;
            end
            
            if obj.save
                obj.G_history{end+1} = fullG;
            end
        end
        %%
        function output = multiply(obj, vec, layer_original)
            % MULTIPLY: forward pass of fully connected / LSTM layers
            % vec (INPUT): 2D matrix, each column per sample
            % layer (INPUT): layer number
            
            % Check is the layer a valid layer and return physical layer no
            layer = obj.check_layer(layer_original);
            
            % Voltage scaling (input * scaling = voltage)
            voltage_scaling = obj.V_read./max(abs(vec));
            voltage_scaling(~any(vec))=1; % Case the whole input is '000...000'
            voltage_scaling_matrix=repmat(voltage_scaling,size(vec,1),1);
          
            V_input=NaN(size(vec).*[2 1]);
            V_input(1:2:end-1,:)=vec.*voltage_scaling_matrix;
            V_input(2:2:end,:)=-V_input(1:2:end-1,:);
            
            V_input = repmat(V_input, obj.dp_rep{layer}(1), 1);
            
            % If the layer is chopped
            if obj.dp_exceed(layer) == 1            
                I_output_chop1 = obj.base.subs{layer}.read_current(V_input(1:end/2,:), 'gain', 2 );
                I_output_chop2 = obj.base.subs{layer}.read_current(V_input(end/2+1:end,:), 'gain', 2);
                I_output = I_output_chop1(1:end/2, :) + I_output_chop2(end/2+1:end, :);
            else
                I_output = obj.base.subs{ layer }.read_current( V_input, 'gain', 2 );                
            end
            
            % From repeated blocks back to averaged single block
            I_output = reshape(I_output, size(I_output,1)/obj.dp_rep{layer}(2), obj.dp_rep{layer}(2), size(I_output,2) );
            I_output = squeeze(mean(I_output, 2));
            
            % Scaling back (voltage and weight scaling)
            output = I_output ./ voltage_scaling / obj.ratio_G_W;
            
            % Plot and save
            if obj.draw >= 2
                obj.plot_IV(V_input, I_output, layer);
            end
            
            if obj.save
                obj.save_IV(V_input, I_output, layer);
            end
        end
        %%
        function output=xcorr3d(obj, input, bias_config, kernel_size, output_dim,...
                strides, layer_original)
            % XCORR3D convolution forward pass hardware call
            % input: Input 4D (height*width*depth*batch_size)
            % bias_config: bias_config (ratio, repeats)
            % kernel_size: 4D (height*width*depth*population)
            % output_dim: 4D (height*width*population*batch_size)
            % strides: 2D (rows / cols) per stride
            % layer_original: the layer no. in the network
            
            % Check is the layer a valid layer and return physical layer no
            layer = obj.check_layer(layer_original);
            
            % Check if there is layer folding
            if obj.dp_exceed(layer) == 1
                error('Conv2d layer does not support folding now.');
            end
            
            % Hardware call
            output = obj.base.subs{ layer }.xcorr3d(obj.dp_rep{layer},...
                input, bias_config, kernel_size, output_dim, strides);
            
            % Scaling back (G based calculation to W based, ratio_G_w)
            output = output / obj.ratio_G_W;
            
            % Plot and save
            if obj.draw >= 2
                obj.plot_IV(input, output, layer);
            end
            
            if obj.save
                obj.save_IV(input, output, layer);
            end
        end
        %%
        function output=xcorr3dLSTM(obj, input_x, input_h, bias_config, ...
                kernel_size_x, kernel_size_h, output_dim, strides_x, layer_original)
            % XCORR3DLSTM convolutional LSTM forward pass hardware call
            % input_x (or h): Input 4D (height*width*depth*batch_size)
            % bias_config: bias_config (ratio, repeats)
            % kernel_size_x (or h): 4D (height*width*depth*population)
            % output_dim: 4D (height*width*population*batch_size)
            % strides_x: 2D (rows / cols) per stride
            % layer_original: the layer no. in the network
            
            % Check is the layer a valid layer and return physical layer no
            layer = obj.check_layer(layer_original);
            
            % Check if there is layer folding
            if obj.dp_exceed(layer) == 1
                error('Conv2d layer does not support folding now.');
            end
            
            % Hardware call
            output = obj.base.subs{ layer }.xcorr3dLSTM(obj.dp_rep{layer},...
                input_x, input_h, bias_config, kernel_size_x,...
                kernel_size_h, output_dim, strides_x);
            
            % Scaling back (G based calculation to W based, ratio_G_w)
            output = output / obj.ratio_G_W;
            
            % Plot and save
            if obj.draw >= 2
                obj.plot_IV(input_x, output, layer);
            end
            
            if obj.save
                input = cell(1,2);
                input{1} = input_x;
                input{2} = input_h;
                obj.save_IV(input, output, layer);
            end
        end
        %%
        function output = multiply_reverse(obj, vec, layer_original)
            % MuLTIPLY_REVERSE: software backpropagation for dense/lstm
            % vec (INPUT) : each col per input vector
            % layer_original : original layer number
            
            % Check is the layer a valid layer and return physical layer no
            layer = obj.check_layer(layer_original);
            
            % Retrieve last read G
            G = obj.array_G{layer};
            
            % If there is layer chopping
            if obj.dp_exceed(layer) == 1
                G = [G(:,1:end/2); G(:,end/2+1:end)];
            end
            
            % Reduce from the vertical differential pair to a single scalar
            w = G(1:2:end-1,:)-G(2:2:end,:);
            
            % From multiple DP back to single DP
            if ~isequal(obj.dp_rep{layer}, [1,1])
                temp = mat2cell(w, ones(1, obj.dp_rep{layer}(1))*size(w,1)/obj.dp_rep{layer}(1),...
                    ones(1, obj.dp_rep{layer}(2))*size(w,2)/obj.dp_rep{layer}(2)); % Break the G into sub-blocks
                temp = reshape(temp(:), 1 , 1, numel(temp));                
                w = mean(cell2mat(temp),3);
            end
            
            % Reverse multiplication
            output = w * vec / obj.ratio_G_W; % w is tranposed compared to upper level algorithrms
        end
        %%
        function output = xcorr3d_reverse(obj, dy, input_dim, pad_size_pre,...
                kernel_size, strides, layer_original)
            % XCORR3D_REVERSE delta backpropagate in CNN layer
            
            % Check is the layer a valid layer and return physical layer no
            layer = obj.check_layer(layer_original);
            
            % Initialize output (dE/dx of the front layer)
            output = zeros([input_dim size(dy,4)]);
            
            % Retrieve last read G
            G = obj.array_G{layer};
            
            % Reduce from the vertical differential pair to a single scalar
            % w = (G(1:2:end-1,:)-G(2:2:end,:)) ; % Slow learning could suppress CNN accuracy loss at large noise
            w = (G(1:2:end-1,:)-G(2:2:end,:)) / obj.ratio_G_W;
            
            % From multiple DP back to single DP
            if ~isequal(obj.dp_rep{layer}, [1,1])
                temp = mat2cell(w, ones(1, obj.dp_rep{layer}(1))*size(w,1)/obj.dp_rep{layer}(1),...
                    ones(1, obj.dp_rep{layer}(2))*size(w,2)/obj.dp_rep{layer}(2)); % Break the G into sub-blocks
                temp = reshape(temp(:), 1 , 1, numel(temp));                
                w = mean(cell2mat(temp),3);
            end
                        
            % 4D kernel based on 2D weight matrix
            temp=w(1:prod(kernel_size(1:3)),:);
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
                            kernel_select = kernel(kernel_row_select, kernel_col_select,depth,:);
                            
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
                kernel_size_x, kernel_size_h, strides_x, layer_original)
            % XCORR3DLSTM_REVERSE delta backpropagate in conv2DLSTM layer
            
            % Check is the layer a valid layer and return physical layer no
            layer = obj.check_layer(layer_original);
            
            % Initialize output (dE/dx of the front layer)
            output_x = zeros([input_dim_x, size(dy,4)]);
            output_h = zeros([input_dim_h, size(dy,4)]);
            
            % Retrieve last read G
            G = obj.array_G{layer};
            
            % Reduce from the vertical differential pair to a single scalar
            % w = (G(1:2:end-1,:)-G(2:2:end,:)) ; % Slow learning could suppress CNN accuracy loss at large noise
            w = (G(1:2:end-1,:)-G(2:2:end,:)) / obj.ratio_G_W;
            
            % From multiple DP back to single DP
            if ~isequal(obj.dp_rep{layer}, [1,1])
                temp = mat2cell(w, ones(1, obj.dp_rep{layer}(1))*size(w,1)/obj.dp_rep{layer}(1),...
                    ones(1, obj.dp_rep{layer}(2))*size(w,2)/obj.dp_rep{layer}(2)); % Break the G into sub-blocks
                temp = reshape(temp(:), 1 , 1, numel(temp));                
                w = mean(cell2mat(temp),3);
            end
            
            % 4D kernel_x and kernel_h based on 2D weight matrix
            temp=w;
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
        %%
        function layer = check_layer(obj, layer_original )
            % CHECK_LAYER: see the input layer number is valid or not
            
            % Check the correspnding
            layer = obj.phys_layer_num(layer_original);
            
            if layer > numel( obj.base.subs )
                error('Layer correpsonds to non-existing physical layer.');
            end
        end
        %%
        function plot_IV(obj, V_input, I_output, layer)
            % Plot physical forward pass input (V) and output (I)
            
            total_layers = max( obj.phys_layer_num(:) );
            
            figure(3);
                       
            if ismatrix(V_input) % Dense Output
            
                subplot(2, total_layers, layer);
                plot(1:size(I_output, 1), I_output , 'o', 'MarkerSize', 1);
                
                title(['I@' num2str(layer) '=' num2str(I_output(1))]);
                grid on; box on;
                %ylim([-2.4e-4 2.4e-4]);
                
                subplot(2, total_layers, layer + total_layers);
                plot(1:size(V_input, 1), V_input , 'o', 'MarkerSize', 1);
                
                title(['V@' num2str(layer) '=' num2str(V_input(1))]);
                grid on; box on;
                ylim([-0.3 0.3]);
                
            else %% Conv2D output
                
                subplot(2, total_layers, layer);
                max_counter = size(V_input, 4);
                V_plot = cell(5, 10);
                counter = 1;
                for row=1:5
                    for col=1:10
                        V_plot{row, col} = V_input(:,:,1,counter);
                        if counter < max_counter                         
                            counter = counter+1;
                        end
                    end
                end
                V_plot = cell2mat(V_plot);
                
                imagesc(V_plot);
                
                subplot(2, total_layers, layer + total_layers);
                size_I = size(I_output);
                I_plot = reshape( I_output, prod(size_I(1:3)), size_I(4));
                plot(I_plot);
                
            end            
            drawnow;
        end
        %%
        function save_IV(obj, V_input, I_output, layer)
            % Save IV data
            
            if layer == 1
                obj.V_vec_history{end+1, layer} = V_input;
                obj.I_history{end+1, layer} = I_output;
            else
                obj.V_vec_history{end, layer} = V_input;
                obj.I_history{end, layer} = I_output;
            end
        end
    end
end