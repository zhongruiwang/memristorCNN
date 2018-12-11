classdef (Abstract) recurrent_layer < layer
    % An abstract class for recurrent layers (LSTM, etc)
    %
    properties (Abstract)
        recur_act_name
    end
end