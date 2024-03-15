import torch 

from backbone.dlanet_dcn import DlaNet



if __name__ == "__main__":
    # model = DlaNet(34, heads={'hm': 1, 'wh': 2, 'ang': 1}, head_conv=256)
    
    # test_tensor = torch.rand(1, 3, 512, 512)
    # output = model(test_tensor)
    # print(output['hm'].size())
    # print(output['wh'].size())
    # print(output['ang'].size())
    # print(len(output)

    # 创建输入张量
    input_tensor = torch.tensor([[1, 2], [3, 4], [5, 6]])

    # 创建索引张量
    index_tensor = torch.tensor([[0, 0], [1, 0], [1, 1]])

    # 在输入张量的第一个维度上使用索引张量进行检索
    output_tensor = torch.gather(input_tensor, dim=0, index=index_tensor)

    print(output_tensor)

    