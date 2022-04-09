function y = CNN_2hiddenlayer(input,kernel1,kernel2,kernel3,kernel4)

   l1 = layer(input,kernel1);
   l2 = layer(l1,kernel2);
   l3 = layer(l2,kernel3);
   y = conv1d(l3,kernel4);

end