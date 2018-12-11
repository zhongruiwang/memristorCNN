classdef cross_entropy_softmax_loss < loss
    % Cross-entropy loss for softmax  
    properties
        calc_mode
        recurrent_mode
    end
    
    methods
        function obj = cross_entropy_softmax_loss( varargin )
            obj = obj@loss( varargin{:} );
        end
        
        function [dys, accuracy]  = calc_delta(obj, ys, ys_train )
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
          
        function loss = calc_loss(obj, ys, ys_train )
            loss = 0;
            
            for t = 1: size(ys, 3)
                %Only the last output matters for some recurrent neural networks
                if strcmp( obj.recurrent_mode, 'last') && t ~= size(ys, 3)
                    continue;
                end
            
                for n = 1: size(ys, 2)
                    for c = 1: size(ys, 1)
                        loss = loss - ys_train( c, n, t) * log( ys(c, n, t) );
                    end
                end
            end
            
            if strcmp( obj.calc_mode, 'mean')
                loss = loss ./  size(ys, 2) ./ size(ys, 3);
            end
        end
    end
end