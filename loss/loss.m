classdef (Abstract) loss < handle
    % An abstract class for loss 
    %%
    properties (Abstract)
        calc_mode % Two modes: sum or mean
        recurrent_mode % Two modes: all or last
    end
    methods
        %%
        function obj = loss( varargin )
            % LOSS abstract function for loss functions
            % INPUT:
            %   calc_mode:      'sum'|'mean' 
            %   recurrent_mode: 'all'|'last'
            %
            % This function is called by the sub-class only.
            
            okargs = {'calc_mode', 'recurrent_mode'};
            defaults = {'sum', 'all'};
            [obj.calc_mode, obj.recurrent_mode] =...
                internal.stats.parseArgs(okargs, defaults, varargin{:});
        end
    end
    
    methods (Abstract)
        
        dys  = calc_delta(~, ys, y_train ); % Output-Label (a batch)
        loss = calc_loss(~, ys, y_train ); % Loss function (a batch)
    end
end