import torch
import torch.nn as nn
import torch.nn.functional as F

class CBR(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(CBR, self).__init__()
        self.op = nn.Sequential(
            nn.Conv1d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(C_out),
            nn.ReLU(inplace=False))

    def forward(self, x):
        return self.op(x)

class DilatedConvNet(nn.Module):
    def __init__(self, EEG_channel, num_classes=2, middle_num=64):
        super(DilatedConvNet, self).__init__()
        
        self.layer1 = nn.Sequential(
            CBR(EEG_channel, EEG_channel * 2, kernel_size=9, stride=4, padding=0),
            CBR(EEG_channel * 2, EEG_channel * 4, kernel_size=3, stride=1, padding=0),
            CBR(EEG_channel * 4, EEG_channel * 4, kernel_size=3, stride=1, padding=0),
        )
        
        self.layer2 = nn.Sequential(
            DilatedConvBlock(EEG_channel * 4, EEG_channel * 4, kernel_size=9, dilation=2),
            DilatedConvBlock(EEG_channel * 4, EEG_channel * 4, kernel_size=3, dilation=2),
            DilatedConvBlock(EEG_channel * 4, EEG_channel * 4, kernel_size=3, dilation=4),
        )
        
        self.residual = nn.Sequential(
            nn.Conv1d(EEG_channel * 4, EEG_channel * 4, kernel_size=1, stride=4),
            nn.BatchNorm1d(EEG_channel * 4)
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.global_pool(x)

        return x

class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(DilatedConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=kernel_size*dilation, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))

#IIMCNet
class IIMCNet(nn.Module):
    def __init__(self, EEG_channel, NIRS_channel,rank,middle_num=64):
        super(IIMCNet, self).__init__()
        self.net12 =DilatedConvNet(EEG_channel+NIRS_channel)
        self.net13 =DilatedConvNet(EEG_channel+NIRS_channel)
        self.net23 =DilatedConvNet(NIRS_channel+NIRS_channel)

        self.EEG_net = nn.Sequential(
            CBR(EEG_channel,EEG_channel*2,kernel_size=9,stride=4,padding=0),
            
            CBR(EEG_channel*2,EEG_channel*2,kernel_size=3,stride=1,padding=0),
            CBR(EEG_channel*2,EEG_channel*2,kernel_size=3,stride=1,padding=0),

            CBR(EEG_channel*2,EEG_channel*4,kernel_size=9,stride=4,padding=0),
            CBR(EEG_channel*4,EEG_channel*4,kernel_size=3,stride=1,padding=0),
            CBR(EEG_channel*4,EEG_channel*4,kernel_size=3,stride=1,padding=0),

            nn.AdaptiveAvgPool1d(1)
        )
        
        self.NIRS_oxy_net = nn.Sequential(
            CBR(NIRS_channel,NIRS_channel*2,kernel_size=5,stride=2,padding=0),
            CBR(NIRS_channel*2,NIRS_channel*2,kernel_size=3,stride=1,padding=0),
            CBR(NIRS_channel*2,NIRS_channel*2,kernel_size=3,stride=1,padding=0),

            CBR(NIRS_channel*2,NIRS_channel*4,kernel_size=5,stride=1,padding=0),
            CBR(NIRS_channel*4,NIRS_channel*4,kernel_size=3,stride=1,padding=0),
            CBR(NIRS_channel*4,NIRS_channel*4,kernel_size=3,stride=1,padding=0),

            nn.AdaptiveAvgPool1d(1)
        )

        self.NIRS_deoxy_net = nn.Sequential(
            CBR(NIRS_channel,NIRS_channel*2,kernel_size=5,stride=2,padding=0),
            CBR(NIRS_channel*2,NIRS_channel*2,kernel_size=3,stride=1,padding=0),
            CBR(NIRS_channel*2,NIRS_channel*2,kernel_size=3,stride=1,padding=0),

            CBR(NIRS_channel*2,NIRS_channel*4,kernel_size=5,stride=1,padding=0),
            CBR(NIRS_channel*4,NIRS_channel*4,kernel_size=3,stride=1,padding=0),
            CBR(NIRS_channel*4,NIRS_channel*4,kernel_size=3,stride=1,padding=0),

            nn.AdaptiveAvgPool1d(1)
        )

        self.out = nn.Sequential(
                nn.Tanh(),
                nn.Linear(408+264+264+288,2),
                nn.Softmax(dim=-1),
            )
        
        self.out1 = nn.Sequential(
                nn.Tanh(),
                nn.Linear(120,2),
                nn.Softmax(dim=-1),
            )
        
        self.out2 = nn.Sequential(
                nn.Tanh(),
                nn.Linear(144,2),
                nn.Softmax(dim=-1),
            )
        
        self.out3 = nn.Sequential(
                nn.Tanh(),
                nn.Linear(144,2),
                nn.Softmax(dim=-1),
            )
        
        self.out12 = nn.Sequential(
                nn.Tanh(),
                nn.Linear(264,2),
                nn.Softmax(dim=-1),
            )
        
        self.out13 = nn.Sequential(
                nn.Tanh(),
                nn.Linear(264,2),
                nn.Softmax(dim=-1),
            )
        
        self.out23 = nn.Sequential(
                nn.Tanh(),
                nn.Linear(288,2),
                nn.Softmax(dim=-1),
            )
 
    def forward(self,EEG_x,NIRS_oxy_x,NIRS_deoxy_x):
        x1=self.EEG_net(EEG_x)
        x1=torch.squeeze(x1)
        x2=self.NIRS_oxy_net(NIRS_oxy_x)
        x2=torch.squeeze(x2)
        x3=self.NIRS_deoxy_net(NIRS_deoxy_x)
        x3=torch.squeeze(x3)

        ##cat ch, interpolate        
        NIRS_oxy_x_int = F.interpolate(NIRS_oxy_x, size=EEG_x.size(2), mode='linear', align_corners=True)
        NIRS_deoxy_int = F.interpolate(NIRS_deoxy_x, size=EEG_x.size(2), mode='linear', align_corners=True)
        x12=self.net12(torch.cat((EEG_x,NIRS_oxy_x_int),dim=1))
        x12=torch.squeeze(x12)
        x13=self.net13(torch.cat((EEG_x,NIRS_deoxy_int),dim=1))
        x13=torch.squeeze(x13)
        x23=self.net23(torch.cat((NIRS_oxy_x_int,NIRS_deoxy_int),dim=1))
        x23=torch.squeeze(x23)
        x=torch.cat([x1,x2,x3,x12,x13,x23],dim=1)
        
        return [self.out(x),self.out1(x1),self.out2(x2),self.out3(x3),self.out12(x12),self.out23(x23),self.out13(x13)]


if __name__=='__main__':
    from thop import profile
    model = IIMCNet(30, 36, 16)
    x1 = torch.randn(4, 30, 600)
    x2 = torch.randn(4, 36, 30)
    x3 = torch.randn(4, 36, 30)
    flops, params = profile(model, inputs=(x1,x2,x3))
    print(flops,params)
