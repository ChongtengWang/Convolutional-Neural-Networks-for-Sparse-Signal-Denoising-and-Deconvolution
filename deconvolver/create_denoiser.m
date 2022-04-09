function y = create_denoiser(length,ratio,threshold1,threshold2,training_type,istrain_flag,training_num,training_sigma)

N = 500;
l = length;
rho = ratio;
t1 = threshold1;
t2 = threshold2;
if istrain_flag

    s1 = 0;
    s2 = 0;
    train_block = training_num;
    sigma = training_sigma;

    test = zeros(train_block, N);
    for i=1:1:train_block
         K = ceil(50*rand()+25);
         index_random = randperm(N);
         index = index_random(1:K);
         if training_type == 1
            test(i,index) = rand(1,K)*20 - 10;
         elseif training_type == 2
            test(i,index) = randn(1,K)*10;
         else 
             sprintf('Warning: wrong input type')
         end
         test(i,:) = test(i,:) + sigma*rand()*randn(1,N);
         positive = ReLU(test(i,:));
         negative = ReLU(-test(i,:));
         s_1 = 0;
         s_2 = 0;
         for j=1+(l-1)/2:1:500-(l-1)/2
             s_1 = s_1 + sum(positive([j-(l-1)/2:j-1,j+1:j+(l-1)/2]));
             s_2 = s_2 + sum(negative([j-(l-1)/2:j-1,j+1:j+(l-1)/2]));
         end
         s1 = s1 + s_1/(501-l);
         s2 = s2 + s_2/(501-l);
    end
    s1 = s1/train_block;
    s2 = s2/train_block;
    
    
else
    
    if training_type == 1
        s1 = 25.5587; 
        s2 = 25.5601;
    elseif training_type == 2
        s1 = 27.5153;
        s2 = 27.5474;
    end
    
end



a1 = rho*t2/(t2-t1);
a2 = rho*t1/(t2-t1);
b1 = -rho*t1*t2/(s1+s2)/(t2-t1);
b2 = -rho*t1*t2/(s1+s2)/(t2-t1);


denoiser1 = zeros(2,1,9);
denoiser1(1,1,5) = 1;
denoiser1(2,1,5) = -1;

denoiser3 = zeros(1,2,9);
denoiser3(1,1,5) = 1;
denoiser3(1,2,5) = -1;

denoiser2 = zeros(2,2,l);
denoiser2(1,1,:) = b1;
denoiser2(1,1,(l+1)/2) = a1;
denoiser2(1,2,:) = b2;
denoiser2(1,2,(l+1)/2) = a2;
denoiser2(2,1,:) = denoiser2(1,2,:);
denoiser2(2,2,:) = denoiser2(1,1,:);


denoiser_CNN{1} = denoiser1;
denoiser_CNN{2} = denoiser2;
denoiser_CNN{3} = denoiser3;
y = denoiser_CNN;
%save('1hiddenlayer_CNN_test2.mat', 'denoiser_CNN');

end
