
仅仅是个人尝试，效果很差，仅供参考，**可能有误**！

1. DataLoad阶段，将预测值Y目标时间，改成和输入时间段长度相同的Y序列，让输出结果拟合趋势（仔细想想好像不太靠谱）。
2. 同时预测了三个输出，对于始终为正的两个输出进行了softplus激活函数，然后对第三个平均，即输出为x,y,z时，处理成$\text{softplus}(x), \text{softplus}(y), (\text{softplus}(x)- \text{softplus}(y)+z)/2$
3. 只是用一个完整模型处理所有数据，其中将product_id进行了embedding，投影到64维的向量上。`torch.nn.Embedding`.
4. 使用了kaiming_normal_初始化。
5. 输入在塞进LSTM之前还可以走一个MLP。
6. LSTM和RNN的差别仅仅在于把`nn.RNN`改成`nn.LSTM`，这……。

不保证正确，结果也很差，**仅供参考**！
如有不同意见**及时反馈**！我也不会