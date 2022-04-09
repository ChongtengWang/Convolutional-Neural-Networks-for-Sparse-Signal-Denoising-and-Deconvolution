function y = CNN_1hiddenlayer(input,kernel1,kernel2,kernel3)

   l1 = layer(input,kernel1);
   l2 = layer(l1,kernel2);
   y = conv1d(l2,kernel3);

end