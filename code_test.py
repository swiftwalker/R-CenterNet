import torch 

from backbone.dlanet_dcn import DlaNet



if __name__ == "__main__":
    model = DlaNet(34, heads={'hm': 1, 'wh': 2, 'ang': 1}, head_conv=256)
    
    test_tensor = torch.rand(1, 3, 512, 512)
    output = model(test_tensor)
    print(output['hm'].size())
    print(output['wh'].size())
    print(output['ang'].size())
    print(len(output))
    