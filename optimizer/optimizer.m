classdef (Abstract) optimizer < handle
    properties (Abstract)
        backend;
    end
    
    methods (Abstract)
        update(obj, grads);
    end
end