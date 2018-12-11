classdef activations
    % This class provides various activation functions

    methods(Static)
        function act = get( act_name, direction)
            % Get the activation function handel. 
            % Input:
            %   act_name:   The name of the activation function, in string
            %   direction:  'act' | 'deriv' 
            %               The function itself | Its derivative
            % Output:
            %   The function handle
            
%             disp([ act_name, '', direction]);
            if strcmp(direction, 'deriv')
                d_str = 'd';
            else
                d_str = '';
            end

            act = str2func(['activations.' d_str act_name]);
        end
        
        % Activation functions
        function y = linear(x)
            y = x;
        end
        
        function y = dlinear(~)
            y = 1;
        end
        
        function y = tanh(x)
            y = tanh(x);
        end
        
        function y = dtanh(x)
            y = 1-x.^2;
        end
        
        function y = sigmoid(x)
            y = 1./(1 + exp(-x));
        end
        
        function y = dsigmoid(x)
            y = x.*(1 - x);
        end
        
        function y = relu(x)
            y = max(0, x);
        end
        
        function y = drelu(x)
            y = x > 0;
        end
        
        function y = softmax(x)
            y = exp(x) ./ sum( exp(x) );
        end
        
        function y = stable_softmax(x)
            y=exp(x-max(x)) ./ sum(exp(x-max(x)));
        end
    end
end