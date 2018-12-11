classdef RMSprop < optimizer
    properties
        
        % RMSprop parameters
        lr = 0;
        momentum = 0;
        decay = 0;
        eps = 0;
        
        backend;
        
        % History of previous dWs and gradient mean square
        dWs_pre;
        grad_mean_sqr
    end
    
    methods
        function obj = RMSprop( varargin )
            % RMSprop constructor function for RMSprop with
            % momentum. 

            okargs = {'lr', 'momentum', 'decay', 'eps'};
            defaults = {0.001, 0, 0.9, 1e-8};
            [obj.lr, obj.momentum, obj.decay, obj.eps] = internal.stats.parseArgs(okargs, defaults, varargin{:});
            
            obj.dWs_pre = {};
            obj.grad_mean_sqr = {};
        end
        
        function update(obj, grads)
            % Input grads: cell array of gradients, each layer per cell
            
            % If it's the first time to use grad_mean_sqr
            if isempty(obj.grad_mean_sqr)
                obj.grad_mean_sqr = cellfun(@(x) zeros(size(x)), grads,'UniformOutput',false);
            end
                
            % If it's the first time to use momentum
            if isempty(obj.dWs_pre)                
                obj.dWs_pre = cellfun(@(x) zeros(size(x)), grads,'UniformOutput',false);
            end
            
            % For all layers
            for l = 1: length(grads)
                
                % Gradient mean square (evolution)
                obj.grad_mean_sqr{l}=obj.decay*obj.grad_mean_sqr{l}+...
                    (1-obj.decay)*grads{l}.^2;
        
                % Add momentum
                obj.dWs_pre{l}= obj.lr * grads{l}./(obj.grad_mean_sqr{l}.^0.5+obj.eps)...
                        -obj.momentum * obj.dWs_pre{l};
                    
            end
            
            % hardware call (although it's dWs_pre, it's not pre at this
            % moment)
            obj.backend.update( obj.dWs_pre );
            
        end
    end
end