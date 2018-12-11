classdef view < handle
    properties
        DEBUG = 0;
        DRAW_PLOT = 0;
        
        data_plot; % Loss
        accuracy_plot; % Accuracy
    end
    
    methods
        function obj = view()
            obj.data_plot = [];
        end
        
        function print(obj, string ) 
            if obj.DEBUG
                disp( string );
            end
        end 
        
        function plot(obj, loss, accuracy)
            
            % Loss save
            obj.data_plot = [obj.data_plot loss];
            
            % Accuracy save
            if isrow(accuracy)
                obj.accuracy_plot = [obj.accuracy_plot; accuracy];
            elseif isrow(accuracy')
                obj.accuracy_plot = [obj.accuracy_plot; accuracy'];
            else
                error('Accuracy dimension weird.');
            end
            
            % Loss and accuracy plot            
            if obj.DRAW_PLOT
                figure(1);
                subplot(1,2,1);
                plot(obj.data_plot);                
                xlabel('No. of updates');
                ylabel('Loss');
                
                subplot(1,2,2);
                plot(obj.accuracy_plot);
                xlabel('No. of updates');
                ylabel('In-minibatch accuracy');
                drawnow;
            end
            
        end
    end
end