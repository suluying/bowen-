import pandas as pd
from scipy import sparse as sp
import pandas as pd
import numpy as np
import torch

import torch.nn as nn
featureDF = pd.read_csv('./data/293triboprotein_enspenstfliter22.csv')#x,y
featureDF = featureDF.dropna()
print(featureDF.shape)
my_test = pd.read_csv('/mnt/md0/luying/ribo/dnabert/DNABERT/examples/sample_data/ft/6/pctrangenetest_all22.tsv', sep='\t')
#my_test = my_test.dropna(subset=['ensp'])
my_test = my_test.reset_index(drop=True)
# 提取my_test中第一列的transcript_id值
my_test_transcript_ids = my_test['0'].values

# 过滤featureDF，只保留transcript_id在my_test第一列的行
featureDF = featureDF[featureDF['transcript_id'].isin(my_test_transcript_ids)]

# 现在filtered_featureDF只包含在my_test第一列出现的transcript_id对应的行
featureDF.shape
#保存all_sequence_outputsnew
#np.save('./data/all_sequence_outputsnew7132.npy', all_sequence_outputsnew)
all_sequence_outputsnew=np.load('./data/all_sequence_outputsnew7132.npy')
all_sequence_outputsnew.shape
#ppi直接加载构建好的npz，以下是构建npz步骤
ppi = pd.read_csv('./data/ppi3ensp.csv')
print(ppi.protein_id2.unique().shape)
gene_id_dict = {gene: i for i, gene in enumerate(featureDF['protein'])}
#将ppi中的Protein1变为protein_id1在gene_id_dict中对应的序号
ppi['Protein1'] = ppi['protein_id1'].map(gene_id_dict)
#将ppi中的Protein2变为protein_id2在gene_id_dict中对应的序号
ppi['Protein2'] = ppi['protein_id2'].map(gene_id_dict)
ppi=ppi.dropna()
#用ppi构建一个coo的稀疏举证，用protein1和protein2构建一个coo的稀疏矩阵
ppi_matrix = sp.coo_matrix((ppi["CombinedScore"], (ppi['Protein1'], ppi['Protein2'])), shape=(featureDF.shape[0], featureDF.shape[0]))

adj=ppi_matrix.todense()
adj.shape
#加入pausingscore

#import pandas as pd
pausing = pd.read_csv('/mnt/md0/luying/ribo/308code/pausing/pause_scores_cdsall.csv')#x,y
#修改pausing的列名为"protein_id","High_Pause_Counts"	“transcript_id”
pausing.columns = ['protein_id',"High_Pause_Counts","transcript_id"]
#将featureDF和pausing合并，使用transcript_id作为合并的键，
merged_df = pd.merge(featureDF, pausing, on='transcript_id', how='left')
print(merged_df)
# 将合并后没有匹配的pause_score值填充为0
merged_df['High_Pause_Counts'].fillna(0, inplace=True)

# 打印合并后的DataFrame
print(merged_df.shape)
from torch_geometric.data import Data
rows = ppi_matrix.row
cols = ppi_matrix.col

# 转换为PyTorch Tensor
# 注意：PyTorch Geometric期望edge_index是一个整数类型的Tensor，通常是torch.long
edge_index = torch.tensor([rows, cols], dtype=torch.long)
x=torch.tensor(merged_df['rNC2'].to_numpy())
# 创建PyTorch Geometric的Data对象
data = Data(x=x, edge_index=edge_index, y=torch.tensor(merged_df['NC3'].to_numpy(), dtype=torch.float32), seq = torch.tensor(all_sequence_outputsnew, dtype=torch.float32),pause = torch.tensor(merged_df['High_Pause_Counts'], dtype=torch.float32))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv  # 使用SAGEConv替代GCNConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
from src.utils import set_seed

# 设置随机种子
SEED = 1
set_seed(SEED)
# 假定 featureDF, ppi_matrix, 和 all_sequence_outputsnew 已经定义好

# 转换为PyTorch Tensor
edge_index = torch.tensor([ppi_matrix.row, ppi_matrix.col], dtype=torch.long)

# 创建PyTorch Geometric的Data对象
train_data = Data(
    x=torch.tensor(merged_df['rNC2'].to_numpy(), dtype=torch.float32),
    edge_index=edge_index,
    y=torch.tensor(merged_df['NC3'].to_numpy(), dtype=torch.float32),
    seq=torch.tensor(all_sequence_outputsnew, dtype=torch.float32),
    pause=torch.tensor(merged_df['High_Pause_Counts'].to_numpy(), dtype=torch.float32)
)

# 创建测试集Data对象
test_data = Data(
    x=torch.tensor(merged_df['rNC1'].to_numpy(), dtype=torch.float32),
    edge_index=edge_index,
    y=torch.tensor(merged_df['NC2'].to_numpy(), dtype=torch.float32),
    seq=torch.tensor(all_sequence_outputsnew, dtype=torch.float32),
    pause=torch.tensor(merged_df['High_Pause_Counts'].to_numpy(), dtype=torch.float32)
)

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 32)
        )

        self.encoder = nn.Sequential(
            nn.Linear(9216, 32),
            nn.GELU(),
        )

        # 替换为SAGEConv
        self.conv1 = SAGEConv(64, 1)

    def forward(self, data):
        x = data.x.view(-1, 1)
        seq_embedding = data.seq
        edge_index = data.edge_index
        pausescore = data.pause.view(-1, 1)

        x = self.fc(x) + self.encoder(seq_embedding)
        x = torch.cat((self.fc(pausescore), x), dim=1)
        x = F.gelu(x)
        x = self.conv1(x, edge_index)  # 传递处理后的特征和边索引

        return x

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
neural_net = NeuralNet().to(device)

# 损失函数和优化器
optimizer = optim.Adam(neural_net.parameters(), lr=0.005)
criterion = nn.MSELoss()

num_epochs = 800000

train_data = train_data.to(device)
test_data = test_data.to(device)

for epoch in range(num_epochs):
    neural_net.train()

    optimizer.zero_grad()

    y_pred = neural_net(train_data).view(-1)
    loss = criterion(y_pred, train_data.y)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        neural_net.eval()
        with torch.no_grad():
            y_train_pred = neural_net(train_data).view(-1).cpu()
            train_r2 = r2_score(y_train_pred.numpy(), train_data.y.cpu().numpy())
            
            y_test_pred = neural_net(test_data).view(-1).cpu()
            test_r2 = r2_score(y_test_pred.numpy(), test_data.y.cpu().numpy())
        
        print(f'Epoch: {epoch}, Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}')
#torch.save(neural_net, './model/bertgraphsagepausing7132_2model.pth')
