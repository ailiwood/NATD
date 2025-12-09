from torch import nn
from models.DCT_Attention import DCT_Attention
from models.EfficientTimeSeriesTransformer import EfficientTimeSeriesTransformer as TransformerBlock


class unsw_nb15_1D(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(unsw_nb15_1D, self).__init__()
        self.enc1 = self.conv_block(input_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)
        self.bottleneck = self.conv_block(256, 512)

        self.up6 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.dec6 = self.conv_block(256, 256)
        self.dct_model_u6 = DCT_Attention(4)

        self.up7 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.dct_model_u7 = DCT_Attention(8)
        self.dec7 = self.conv_block(128, 128)

        self.dct_model_u8 = DCT_Attention(16)
        self.up8 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.u8_transformer = TransformerBlock(64)  # Transformer Block
        self.dec8 = self.conv_block(64, 64)

        self.dct_model_u9 = DCT_Attention(32)
        self.up9 = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)
        self.u9_transformer = TransformerBlock(32)  # Transformer Block
        self.dec9 = self.conv_block(32, 32)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1024, 64)
        self.fc2 = nn.Linear(64, 16)
        self.output = nn.Linear(16, num_classes)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = nn.MaxPool1d(2)(e1)
        e2 = self.enc2(p1)
        p2 = nn.MaxPool1d(2)(e2)
        e3 = self.enc3(p2)
        p3 = nn.MaxPool1d(2)(e3)
        e4 = self.enc4(p3)
        p4 = nn.MaxPool1d(2)(e4)

        bn = self.bottleneck(p4)

        u6 = self.up6(bn)
        u6_residual = e4[:, :, :4]
        u6 = self.dct_model_u6(u6[:, :, :4])
        u6 = u6 + u6_residual
        d6 = self.dec6(u6)

        u7 = self.up7(d6)
        u7_residual = e3[:, :, :8]
        u7 = self.dct_model_u7(u7[:, :, :8])
        u7 = u7 + u7_residual
        d7 = self.dec7(u7)

        u8 = self.up8(d7)
        u8_residual = e2[:, :, :16]
        u8 = self.dct_model_u8(u8[:, :, :16])
        u8 = u8 + u8_residual
        u8 = self.u8_transformer(u8)
        d8 = self.dec8(u8)

        u9 = self.up9(d8)
        u9_residual = e1[:, :, :32]
        u9 = self.dct_model_u9(u9[:, :, :32])
        u9 = u9 + u9_residual
        u9 = self.u9_transformer(u9)
        d9 = self.dec9(u9)

        out = self.flatten(d9)
        out = self.fc1(out)
        out = self.fc2(out)
        return self.output(out)




