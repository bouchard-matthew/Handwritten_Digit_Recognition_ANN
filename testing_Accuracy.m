%Uncomment the confusion matrix needed to calculate for:

%Training input - 80% of the training data; models{val}, val corresponds to
%... network of which you want to calculate the accuracy of this data set
%... for
[c,cm,ind,per] = confusion(y1,models{1}(x1));

%Training input - 20% of the training data; models{val}, val corresponds to
%... network of which you want to calculate the accuracy of this data set
%... for
%[c,cm,ind,per] = confusion(y2d,models{1}(x2));

%Testing input; models{val}, val corresponds to
%... network of which you want to calculate the accuracy of this data set
%... for
%[c,cm,ind,per] = confusion(targetsd_test,models{1}(inputs_test));

%True Positive, True Negative, False Positive, False Negative
TP = per(:,3);
TN = per(:,4);
FP = per(:,2);
FN = per(:,1);

%Total counter variables for tp, tn, fp, fn
tp_total = zeros(1);
tn_total = zeros(1);
fp_total = zeros(1);
fn_total = zeros(1);
for i = 1:10
    tp_total = tp_total + TP(i);
    tn_total = tn_total + TN(i);
    fp_total = fp_total + FP(i);
    fn_total = fn_total + FN(i);
end

%(TP + TN) / (TP + TN + FP + FN) = Accuracy of the data set
acc = (tp_total + tn_total)/(tp_total + tn_total + fp_total + fn_total);