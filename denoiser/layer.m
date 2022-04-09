function y = layer(input,kernel)

   y = conv1d(input,kernel);
   y = ReLU(y);

end
