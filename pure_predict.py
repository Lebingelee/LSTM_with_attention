import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import datapreprocess 
import rnn_model
import tool
import pandas as pd

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device('mps')


train_batch_size = 2048
def DataLoad(df_data,seq_length):
    global train_batch_size
    X = scaler.fit_transform(df_data[feature].values)
    # Time series
    X_seq = [torch.tensor(X[i:i+seq_length], dtype=torch.float32) for i in range(len(X) - seq_length)]

    # y_apply = df_data['apply_amt'].values[seq_length:]
    # y_redeem = df_data['redeem_amt'].values[seq_length:]
    # y_net = df_data['net_in_amt'].values[seq_length:]

    # change to sequence
    y_apply = torch.stack([torch.tensor(df_data['apply_amt'].values[i:i+seq_length], dtype=torch.float32) for i in range(len(X) - seq_length)], dim=0).view(-1, seq_length, 1).to(device)
    y_redeem = torch.stack([torch.tensor(df_data['redeem_amt'].values[i:i+seq_length], dtype=torch.float32) for i in range(len(X) - seq_length)], dim=0).view(-1, seq_length, 1).to(device)
    y_net = torch.stack([torch.tensor(df_data['net_in_amt'].values[i:i+seq_length], dtype=torch.float32) for i in range(len(X) - seq_length)], dim=0).view(-1, seq_length, 1).to(device)
    
    
    y = torch.cat([y_apply, y_redeem, y_net], dim=2)

    # 对product_id处理，变成long
    product_id = df_data['product_pid'].values[seq_length:]
    # procuct_id的格式是productxxx，
    product_id = [int(pid[7:]) for pid in product_id]
    product_id = torch.tensor(product_id, dtype=torch.long).to(device)
    dataset = TensorDataset(torch.stack(X_seq), y, product_id)
    return DataLoader(dataset, batch_size=train_batch_size, shuffle=True)
    # Dataset and dataloader
    # dataset_apply = TensorDataset(torch.stack(X_seq), y_apply)
    # dataset_redeem = TensorDataset(torch.stack(X_seq), y_redeem)
    # dataset_net = TensorDataset(torch.stack(X_seq), y_net)
    # dataset = TensorDataset(torch.stack(X_seq), y)
    # return DataLoader(dataset, batch_size=32, shuffle=True)
    
    # apply = DataLoader(dataset_apply, batch_size=32, shuffle=True)
    # redeem = DataLoader(dataset_redeem, batch_size=32, shuffle=True)
    # net = DataLoader(dataset_net, batch_size=32, shuffle=True)
    # return apply,redeem,net

if __name__=="__main__":
    # Data load
    train_path = './data/product_info_simple_final_train.csv'
    test_path = './data/predict_table.csv'
    yield_path = './data/cbyieldcurve_info_final.csv'
    time_path = './data/time_info_final.csv'

    df_raw = datapreprocess.data_load(train_path,test_path,yield_path,time_path)

    # Data process
    feature = ['days_since_last_trade_date', 'days_since_next_trade_date', 'day_of_week', 'is_week_end', 
                'is_wknd', 'open_days', 'days_since_first_transaction']
    label = ['apply_amt', 'redeem_amt', 'net_in_amt']
    
    # df_train, df_test, df_val = datapreprocess.data_process(df_raw)
    df_train, df_test = datapreprocess.data_process_no_val(df_raw)
    print(len(df_train), len(df_test))
    Train = tool.CSV(df_train)
    Test = tool.CSV(df_test)
    # print(len(df_train), len(df_test), len(df_val))
    # Train = tool.CSV(df_train)
    # Test = tool.CSV(df_test)
    # Val = tool.CSV(df_val)

    # Parameters
    seq_length = 12         # sequence length
    hidden_dim = 128
    output_dim = 3
    layers = 4               # RNN layers
    n_epochs = 500
    learning_rate = 0.003

    # remove product which numbers less than 30
    # Val.rm_product_smaller_nums(seq_length)
    # df_val = Val.csv

    product_id_num = len(Train.product_ID) * 4
    print(f'product_id_num: {product_id_num}')
    model = rnn_model.RNN_with_product_id_embedding(input_dim=len(feature), hidden_dim=hidden_dim, output_dim=output_dim, layers=layers, product_id_num=product_id_num).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.003)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
    loss_func = nn.MSELoss()
    scaler = MinMaxScaler()
    
    # 根据model的输入输出创建数据集dataloader
    train_loader = DataLoad(df_train,seq_length)
    test_loader = DataLoad(df_test,seq_length)
    # val_loader = DataLoad(df_val,seq_length)
    

    # Train

    pbar = tqdm(range(n_epochs))
    gamma = 5.0

    model.load_state_dict(torch.load('./model.pth'))
    
    with torch.no_grad():
        # apply to df_test
        model.eval()
        
        
        # read directly from df_test
        X = scaler.fit_transform(df_test[feature].values)
        product_id = df_test['product_pid'].values
        # procuct_id的格式是productxxx，
        product_id = [int(pid[7:]) for pid in product_id]
        product_id = torch.tensor(product_id, dtype=torch.long).to(device)
        # split X according to product_id
        unique_product_ids = torch.unique(product_id)
        # predictions = []

        for pid in unique_product_ids:
            indices = (product_id == pid).nonzero(as_tuple=True)[0].detach().cpu()
            X_pid = torch.tensor(X[indices.detach().cpu()], dtype=torch.float32).to(device)
            X_pid = X_pid.view(1, X_pid.size(0), X_pid.size(1))
            pid = pid.view(1)
            output = model.forward_all(X_pid, pid)
            
            df_test.loc[indices, 'apply_amt_pred'] = output[0, :, 0].cpu().numpy()
            df_test.loc[indices, 'redeem_amt_pred'] = output[0, :, 1].cpu().numpy()
            df_test.loc[indices, 'net_in_amt_pred'] = output[0, :, 2].cpu().numpy()
            
    # 5. 后处理和保存结果
    submit = df_test[['product_pid', 'transaction_date', 'apply_amt_pred', 'redeem_amt_pred', 'net_in_amt_pred']]
    # submit.loc[submit['apply_amt_pred'] < 0, 'apply_amt_pred'] = 0
    # submit.loc[submit['redeem_amt_pred'] < 0, 'redeem_amt_pred'] = 0
    # submit.loc[:, 'net_in_amt_pred'] = submit['net_in_amt_pred'] * 0.5 + 0.5 * (submit['apply_amt_pred'] - submit['redeem_amt_pred'])

    submit.to_csv('./predict_table.csv', index=None)

    # save model
    # torch.save(model.state_dict(), './model.pth')