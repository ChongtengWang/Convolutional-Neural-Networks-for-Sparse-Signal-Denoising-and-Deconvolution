function y = CNN(input, denoiser_CNN)

   if length(denoiser_CNN) == 3
      y = CNN_1hiddenlayer(input, denoiser_CNN{1},denoiser_CNN{2},denoiser_CNN{3});
   elseif length(denoiser_CNN) == 4
           y = CNN_2hiddenlayer(input, denoiser_CNN{1},denoiser_CNN{2},denoiser_CNN{3},denoiser_CNN{4});
   elseif length(denoiser_CNN) == 5
           y = CNN_3hiddenlayer(input, denoiser_CNN{1},denoiser_CNN{2},denoiser_CNN{3},denoiser_CNN{4},denoiser_CNN{5});
   end

end