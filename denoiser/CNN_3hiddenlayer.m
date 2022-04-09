function y = CNN_3hiddenlayer(input,kernel1,kernel2,kernel3,kernel4, kernel5)

   l1 = layer(input,kernel1);
   l2 = layer(l1,kernel2);
   l3 = layer(l2,kernel3);
   l4 = layer(l3,kernel4);
   y = conv1d(l4,kernel5);

end