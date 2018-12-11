classdef mean_square_loss < loss
    % Mean squre loss    
    properties
        calc_mode
        recurrent_mode
    end
    
    methods
        function obj = mean_square_loss( varargin )
            obj = obj@loss( varargin{:} );
        end
        
        function [dys, accuracy]  = calc_delta(obj, ys, ys_train )
            % CALC_DELTA returns the difference between y_train and ys
            % It is called by calc_loss only?
            dys = ys_train - ys;
            
            % Only the last output matters for some recurrent neural networks
            if strcmp( obj.recurrent_mode, 'last')
                dys(:,:,1:end-1) = 0;
                [~, winner_ys] = max(ys(:,:,end));
                [~, winner_ys_train] = max(ys_train(:,:,end));
            else
                [~, winner_ys] = max(ys);
                [~, winner_ys_train] = max(ys_train);
            end
            
            accuracy = mean(winner_ys(:) == winner_ys_train(:));
        end
        
        function loss = calc_loss(obj, ys, y_train )
            % CALC_LOSS returns the loss
            
            loss = 0;
            
            dys = obj.calc_delta( ys, y_train);
            
            for t = 1: size(ys, 3)
                %Only output matters for some recurrent neural networks
                if strcmp( obj.recurrent_mode, 'last') && t ~= size(ys, 3)
                    continue;
                end
            
                for n = 1: size(ys, 2)
                    loss = loss + dys(:, n, t)' * dys(:, n, t);
                end
            end
            
            if strcmp( obj.calc_mode, 'mean')
                loss = loss ./  size(ys, 2) ./ size(ys, 3);
            end
        end
    end
end