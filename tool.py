import numpy as np
import matplotlib.pyplot as plt
import torch
''' 
csv's parameter
@parament len is the different product_pid's number
@parament product_ID is the different product_pid's list
'''
class CSV:
    def __init__(self,csv):
        self.csv = csv
        self.len, self.product_ID = self.get_len()

    def get_len(self):
        product_ID = set()
        for id in self.csv['product_pid']:
            product_ID.add(id)
        return len(product_ID),product_ID
    
    def get_product_id(self):
        product_ID = {}
        for id in self.csv['product_pid']:
            if product_ID.get(id) == None:
                product_ID[id]=1
            else:
                product_ID[id] += 1
        return product_ID
    
    def rm_product_smaller_nums(self,nums):
        product_ID_list = self.get_product_id()
        for key,value in product_ID_list.items():
            if value <= nums:
                self.csv = self.csv[self.csv['product_pid']!=key].reset_index(drop=True)

    

'''
Weighted Mean Absolute Percentage Erroe
'''
def WMAPE(predictions,targets):
    epsilon = 1e-8
    targets_abs = torch.abs(targets) + epsilon

    # 计算权重（实际值的绝对值作为权重）
    weights = targets_abs

    # 计算误差比例
    error_ratio = torch.abs(predictions - targets) / targets_abs

    # 计算加权误差
    weighted_error = torch.sum(weights * error_ratio)

    # 计算总权重
    total_weight = torch.sum(weights)

    # 计算最终的 WMAPE
    wmape = weighted_error / total_weight
    return wmape

'''
Draw WMAPE with epoch
'''
def draw_wmape(wmape,epochs):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), wmape, label='Val WMAPE')
    plt.xlabel('Epoch')
    plt.ylabel('Wmape')
    plt.title('Validation Wmape')
    plt.legend()
    plt.grid()
    plt.savefig('./val_wmape.png')