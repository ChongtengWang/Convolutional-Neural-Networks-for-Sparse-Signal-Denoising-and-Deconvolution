function y = CNN(input, denoiser_CNN)

    l = cell(1,length(denoiser_CNN));
    l{1} = input;
    
    for i = 1:1:length(denoiser_CNN)-1
        
        l{i+1} = layer(l{i},denoiser_CNN{i});
        
    end
    
    y = conv1d(l{length(denoiser_CNN)},denoiser_CNN{length(denoiser_CNN)});

end