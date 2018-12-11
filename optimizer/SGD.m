classdef SGD < optimizer
    properties
        lr = 0;
        momentum = 0;
        
        backend;
        dWs_pre;
    end
    
    methods
        function obj = SGD( varargin )
            % SGD constructor function for stochasitc gradient descent with
            % momentum. Will add learning rate decay later.
            %
            okargs = {'lr', 'momentum'};
            defaults = {0.1, 0};
            [obj.lr, obj.momentum] = internal.stats.parseArgs(okargs, defaults, varargin{:});
            
            obj.dWs_pre = {};
        end
        
        function update(obj, grads)
            % Enforce the learning rate
            % TODO: check momentum, it is NOT working yet!!
            %
            dWs = cellfun(@(x) x.*obj.lr, grads,'UniformOutput',false);
            
            if ~isempty( obj.dWs_pre)
                for l = 1: length(grads)
                    % Add momentum
                    dWs{l} = dWs{l} + obj.dWs_pre{l} .* obj.momentum;
                end
            end
            
            % hardware call
            obj.backend.update( dWs );
            
            % store the weight updates for future use
            obj.dWs_pre = dWs;
        end
            
    end
end