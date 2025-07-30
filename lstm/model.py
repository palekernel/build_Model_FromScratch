from pyexpat import model
import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim) -> None:
        """
        LSTM单元初始化
        参数:
            input_dim: 输入特征的维度
            hidden_dim: 隐藏状态的维度
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 遗忘门 - 控制哪些信息从细胞状态中被遗忘
        self.forget_gate = nn.Linear(self.input_dim + self.hidden_dim, self.hidden_dim)
        # 输入门 - 控制哪些新信息被存入细胞状态
        self.input_gate = nn.Linear(self.input_dim + self.hidden_dim, self.hidden_dim)
        # 候选细胞状态 - 生成新的候选值用于更新细胞状态
        self.cell_gate = nn.Linear(self.input_dim + self.hidden_dim, self.hidden_dim)
        # 输出门 - 控制哪些信息从细胞状态输出到隐藏状态
        self.output_gate = nn.Linear(self.input_dim + self.hidden_dim, self.hidden_dim)

    def forward(self, x, states=None):
        """
        前向传播
        参数:
            x: 输入张量，形状为(batch_size, input_dim)
            states: 包含(hidden_state, cell_state)的元组，每个形状为(batch_size, hidden_dim)
        返回:
            元组(new_hidden_state, new_cell_state)
        """
        if states is None:
            # 如果没有提供初始状态，则初始化为零
            batch_size = x.size(0)
            hidden_state = torch.zeros(batch_size, self.hidden_dim).to(x.device)
            cell_state = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        else:
            hidden_state, cell_state = states
        
        # 拼接输入和隐藏状态
        combined = torch.cat((x, hidden_state), dim=1)
        # 计算遗忘门
        forget = torch.sigmoid(self.forget_gate(combined))
        # 计算输入门和候选细胞状态
        input_gate = torch.sigmoid(self.input_gate(combined))
        candidate_cell = torch.tanh(self.cell_gate(combined))
        # 更新细胞状态
        new_cell_state = forget * cell_state + input_gate * candidate_cell
        # 计算输出门和新隐藏状态
        output_gate = torch.sigmoid(self.output_gate(combined))
        new_hidden_state = output_gate * torch.tanh(new_cell_state)
        return new_hidden_state, new_cell_state

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_players) -> None:
        """
        多玩家LSTM模型
        
        参数:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            num_players: 玩家数量
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_players = num_players  # 修正变量名
        
        # 为每个玩家创建独立的LSTM单元
        self.lstm_cells = nn.ModuleList(
            [LSTMCell(input_dim, hidden_dim) for _ in range(num_players)]
        )
        # 可学习的初始状态
        self.init_h = nn.Parameter(torch.zeros(1, hidden_dim))
        self.init_c = nn.Parameter(torch.zeros(1, hidden_dim))

    def forward(self, x, init_states=None):
        """
        前向传播
        
        参数:
            x: 输入张量，形状为(batch_size, num_players, input_dim)
            init_states: 初始状态元组(h, c)，每个形状为(num_players, batch_size, hidden_dim)
            
        返回:
            out: 所有玩家的输出，形状为(num_players, batch_size, hidden_dim)
            (h, c): 最终状态元组
        """
        batch_size = x.size(0)
        
        # 初始化状态
        if init_states is None:
            # 扩展初始状态到合适的形状
            h = self.init_h.expand(batch_size, -1).unsqueeze(0).expand(self.num_players, -1, -1)
            c = self.init_c.expand(batch_size, -1).unsqueeze(0).expand(self.num_players, -1, -1)
        else:
            h, c = init_states
        
        # 存储所有玩家的输出
        outputs = []
        
        # 处理每个玩家的输入
        for i in range(self.num_players):
            player_input = x[:, i, :]
            # 使用临时变量存储新状态
            new_h, new_c = self.lstm_cells[i](player_input, (h[i], c[i]))
            # 创建新的h和c张量
            h = torch.cat([h[:i], new_h.unsqueeze(0), h[i+1:]])
            c = torch.cat([c[:i], new_c.unsqueeze(0), c[i+1:]])
            outputs.append(new_h.unsqueeze(1))
        
        # 拼接所有输出 (batch_size, num_players, hidden_dim)
        outputs = torch.cat(outputs, dim=1)
        
        return outputs, (h, c)

class lstm(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_players) -> None:
        super().__init__()
        self.lstm = LSTM(input_dim, hidden_dim, num_players)
    def forward(self, x, init_states=None):
        outputs, (h, c) = self.lstm(x, init_states)
        return outputs, (h, c)

if __name__ == '__main__':

    model=LSTM(10,20,3)
    x=torch.randn(5,3,10)
    outputs,_ = model(x)
    print(outputs.shape)

  

