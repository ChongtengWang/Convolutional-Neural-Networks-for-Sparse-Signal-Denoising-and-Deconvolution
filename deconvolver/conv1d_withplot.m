function y = conv1d_withplot(input, kernel)

  l = size(input,2);
  [co,ci,~] = size(kernel);
  y = zeros(co,l);
  inter = zeros(ci,l);
  for i=1:1:co
      figure()
      for j=1:1:ci
          inter(j,:) = conv(input(j,:),flip(squeeze(kernel(i,j,:))'),'same');
          subplot(ci+2,1,j)
          plot(inter(j,:))
          title(['output channel ',num2str(i),', input channel ',num2str(j)]);
      end
      y(i,:) = sum(inter,1);
      subplot(ci+2,1,ci+1)
      plot(y(i,:))
      title(['output channel ',num2str(i),' summation']);
      subplot(ci+2,1,ci+2)
      plot(ReLU(y(i,:)))
      title(['output channel ',num2str(i),' summation after ReLU']);
  end

end