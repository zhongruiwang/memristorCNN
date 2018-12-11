classdef LSTM < recurrent_layer
    % This class provides various activation functions
    properties
        nlayer      % current layer number
        backend
        
        bias_config % a two dimensional array: [ratio rep]
        
        % dimensions
        input_dim
        output_dim
        weight_dim
        
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
        x_in_history
            
        ha_history
        hi_history
        hf_history
        ho_history
        
        hc_history
        hc_pre_history
%         y_out_history
    end
    
    methods
        function obj = LSTM( output_dim, varargin )
            % LSTM the construction funciton for a LSTM layer.
            % 
            % Will need to write a recurrent layer base layer in the
            % future.
            %
            
            okargs = {'input_dim', 'activation', 'recurrent_activation', 'bias_config'};
            defaults = {NaN, 'tanh', 'sigmoid', [1 1]};
            [obj.input_dim, obj.act_name, obj.recur_act_name, obj.bias_config] =...
                internal.stats.parseArgs(okargs, defaults, varargin{:});
            
            obj.output_dim = output_dim;
            obj.set_weight_dim();
            
            obj.initialize();
        end
        
        function initialize(obj)
            obj.y_pre = [];
            obj.hc_pre = [];
            
            obj.dy_next = zeros( obj.output_dim, 1);
            obj.dhc_next = zeros( obj.output_dim, 1);
            
            obj.x_in_history = [];
            
            obj.ha_history = [];
            obj.hi_history = [];
            obj.hf_history = [];
            obj.ho_history = [];

            obj.hc_history = [];
            obj.hc_pre_history = [];
        end
        
        function set_weight_dim(obj)
            obj.weight_dim = [obj.output_dim * 4 ...
                obj.input_dim + obj.output_dim + obj.bias_config(2) ];
        end
       
        function y_out = call(obj, x_in)
            % CALL The forward pass of the layer
            % Input:
            %   x_in:   The input of the layer
            % Output:
            %   y_out:  The output
            %
            n = size(x_in, 2); % The batch size
            
            % Time zero
            if isempty( obj.y_pre )
                obj.y_pre = zeros( obj.output_dim, n);
                obj.hc_pre = zeros( obj.output_dim, n);
            end
            
            x_in_full = [x_in; obj.y_pre; ...
                repmat( obj.bias_config(1), obj.bias_config(2), n)];
            
            % HARDWARE call
            z = obj.backend.multiply( x_in_full, obj.nlayer );
            
            % LSTM activations
            act = activations.get( obj.act_name, 'act');
            recur_act = activations.get( obj.recur_act_name, 'act');
            
            ha =       act( z(                     1 : 1 * obj.output_dim, :) );
            hi = recur_act( z(1 * obj.output_dim + 1 : 2 * obj.output_dim, :) );
            hf = recur_act( z(2 * obj.output_dim + 1 : 3 * obj.output_dim, :) );
            ho = recur_act( z(3 * obj.output_dim + 1 : 4 * obj.output_dim, :) );
            
            hc = hi .* ha + hf .* obj.hc_pre;
            y_out  = ho .* act(hc);
            
            % Store the values after activation for back propogation
            
            obj.x_in_history = cat(3, obj.x_in_history, x_in_full);
            
            obj.ha_history = cat(3, obj.ha_history, ha);
            obj.hi_history = cat(3, obj.hi_history, hi);
            obj.hf_history = cat(3, obj.hf_history, hf);
            obj.ho_history = cat(3, obj.ho_history, ho);
            
            obj.hc_history = cat(3, obj.hc_history, act(hc));
            obj.hc_pre_history = cat(3, obj.hc_pre_history, obj.hc_pre);
            
            % Recurrent inputs
            obj.y_pre = y_out;
            obj.hc_pre = hc;
        end
        
        function [grads, dx] = calc_gradients(obj, dy)
            % CALC_GRADIENTS: The backward pass of the layer
            % Input:
            %   dy:     The delta on this layer
            % Output:
            %   dW:     The weight gradient in this layer
            %   dx:     The delta for previous layer
            %
            
            % Calculate the delta before activation
%             [y_out,  obj.y_out_history] = obj.history_pop( obj.y_out_history);
            [hc,  obj.hc_history] = obj.history_pop( obj.hc_history);
            [hc_pre_,  obj.hc_pre_history] = obj.history_pop( obj.hc_pre_history);
            
            [ha,  obj.ha_history] = obj.history_pop( obj.ha_history);
            [hi,  obj.hi_history] = obj.history_pop( obj.hi_history);
            [hf,  obj.hf_history] = obj.history_pop( obj.hf_history);
            [ho,  obj.ho_history] = obj.history_pop( obj.ho_history);
            [x_in,  obj.x_in_history] = obj.history_pop( obj.x_in_history);
            
            % LSTM activations
            act = activations.get( obj.act_name, 'deriv');
            recur_act = activations.get( obj.recur_act_name, 'deriv');
            
            dy = dy + obj.dy_next; % On
            
            dho = dy .* hc .* recur_act( ho );
            dhc = dy .* ho .* act(hc) + obj.dhc_next;
            
            dha = dhc .* hi .* act(ha);
            dhi = dhc .* ha .* recur_act(hi);
            dhf = dhc .* hc_pre_ .* recur_act(hf);
            
            obj.dhc_next = dhc .* hf;
            
            dz = [dha; dhi; dhf; dho];
            
            % HARDWARE call (backpropogation)
            dx = obj.backend.multiply_reverse( dz, obj.nlayer);
            % store the recurrent delta
            obj.dy_next = dx( obj.input_dim + 1: obj.input_dim + obj.output_dim, :); 
            % Previous layer delta
            dx = dx(1: obj.input_dim);

            grads = dz * x_in.';
        end
    end
end