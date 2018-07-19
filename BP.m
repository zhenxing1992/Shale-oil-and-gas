clear all
traindata=xlsread('Data/train_coal.xlsx')'
train_data=traindata(1:4, :)
train_toc=traindata(5, :)


testdata=xlsread('Data/test_coal.xlsx')'
test_data=testdata(1:4, :)
test_toc=testdata(5, :)


for i=1:3
    var_train=var(train_data(i, :))
    mean_train=mean(train_data(i,:))
    train_data(i,:)=(train_data(i,:)-mean_train)/var_train
end

for i=1:3
    var_test=var(test_data(i, :))
    mean_test=mean(test_data(i,:))
    test_data(i, :)=(test_data(i, :)-mean_test)/(var_test)
end

