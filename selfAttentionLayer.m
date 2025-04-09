classdef selfAttentionLayer < nnet.layer.Layer
    % selfAttentionLayer layer.

    properties (Learnable)
        % Layer learnable parameters
            
        % Scaling coefficient
        Alpha
    end
    
    methods
        function layer = selfAttentionLayer(numChannels, name) 
            % creates a selfAttentionLayer layer
            % for 2-D image input with numChannels channels and specifies 
%             the layer name.

            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = "SelfAttention with " + numChannels + " channels";
        
            % Initialize scaling coefficient.
            layer.Alpha = rand([1 1 numChannels]); 
        end
        
        function Z = predict(layer, X)
            % Z = predict(layer, X) forwards the input data X through the
            % layer and outputs the result Z.
            
            Z = max(X,0) + layer.Alpha .* min(0,X);
        end
    end
end