function y = conv1d(input, kernel)

  l = size(input,2);
  [co,ci,~] = size(kernel);
  y = zeros(co,l);
  inter = zeros(ci,l);
  for i=1:1:co
      for j=1:1:ci
          inter(j,:) = conv(input(j,:),flip(squeeze(kernel(i,j,:))'),'same');
      end
      y(i,:) = sum(inter,1);
  end

end
       