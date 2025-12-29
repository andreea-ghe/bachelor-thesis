import torch
import torch.nn as nn
from .utils_encoder_decoder import PointNetEncoder, PointNetDecoder


class PointNetPTMSG(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.en1 = PointNetEncoder(0.15, [0.05, 0.1], [16, 32], self.in_channels,
                                                    [[16, 16, 32], [32, 32, 64]])
        self.en2 = PointNetEncoder(0.25, [0.1, 0.2], [16, 32], 32 + 64, 
                                                    [[64, 64, 128], [64, 96, 128]])
        self.en3 = PointNetEncoder(0.25, [0.2, 0.4], [16, 32], 128 + 128,
                                                    [[128, 196, 256], [128, 196, 256]])
        self.en4 = PointNetEncoder(0.25, [0.4, 0.8], [16, 32], 256 + 256,
                                                    [[256, 256, 512], [256, 384, 512]])
        self.de4 = PointNetDecoder(512 + 512 + 256 + 256, [256, 256])
        self.de3 = PointNetDecoder(128 + 128 + 256, [256, 256])
        self.de2 = PointNetDecoder(32 + 64 + 256, [256, 128])
        self.de1 = PointNetDecoder(128 + 3, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, self.out_channels, 1)

    def forward(self, x, batch_length):
        """
        Input:
            x represents all the points from all pieces concatenated [N_sum, 3] 
            batch_length = [B]
        """

        idx = 0
        pieces_ids = []
        for i in range(len(batch_length)):
            pieces_ids.append(idx * torch.ones(batch_length[i], dtype=torch.int64).to(x.device))
            idx += 1
        pieces_ids = torch.cat(pieces_ids).reshape(1, 1, -1)  # [1, 1, N_sum]

        layer0_points = x.unsqueeze(0).permute(0, 2, 1)  # [1, 3, N_sum]
        layer0_xyz = x[:, :3].unsqueeze(0).permute(0, 2, 1)  # [1, 3, N_sum]
        layer0_piece_id = pieces_ids  # [1, 1, N_sum]

        layer1_xyz, layer1_piece_id, layer1_points = self.en1(layer0_xyz, layer0_points, layer0_piece_id)
        layer2_xyz, layer2_piece_id, layer2_points = self.en2(layer1_xyz, layer1_points, layer1_piece_id)
        layer3_xyz, layer3_piece_id, layer3_points = self.en3(layer2_xyz, layer2_points, layer2_piece_id)
        layer4_xyz, layer4_piece_id, layer4_points = self.en4(layer3_xyz, layer3_points, layer3_piece_id)

        layer3_points = self.de4(layer3_xyz, layer4_xyz, layer3_piece_id, layer4_piece_id, layer3_points, layer4_points)
        layer2_points = self.de3(layer2_xyz, layer3_xyz, layer2_piece_id, layer3_piece_id, layer2_points, layer3_points)
        layer1_points = self.de2(layer1_xyz, layer2_xyz, layer1_piece_id, layer2_piece_id, layer1_points, layer2_points)
        layer0_points = self.de1(layer0_xyz, layer1_xyz, layer0_piece_id, layer1_piece_id, layer0_points, layer1_points)

        feat = self.conv1(layer0_points)  # [1, feat_out, N_sum]
        feat = feat.permute(0, 2, 1).squeeze(0)  # [N_sum, feat_out]

        return feat
    
if __name__ == '__main__':
    xx = torch.randn(100, 3)
    bl = torch.tensor([20, 20, 35, 25], dtype=torch.long).reshape(4, 1)
    model_point = PointNetPTMSG(3, 128)
    feat_point = model_point(xx, bl)
    print(feat_point.shape)
