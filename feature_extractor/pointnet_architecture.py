import torch
import torch.nn as nn
from .utils_encoder_decoder import PointNetEncoder, PointNetDecoder


class PointNetPTMSG(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Layer names must match checkpoint: sa1-4 for set abstraction, fp1-4 for feature propagation
        self.sa1 = PointNetEncoder(0.15, [0.05, 0.1], [16, 32], self.in_channels,
                                   [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetEncoder(0.25, [0.1, 0.2], [16, 32], 32 + 64,
                                   [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetEncoder(0.25, [0.2, 0.4], [16, 32], 128 + 128,
                                   [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetEncoder(0.25, [0.4, 0.8], [16, 32], 256 + 256,
                                   [[256, 256, 512], [256, 384, 512]])
        self.fp4 = PointNetDecoder(512 + 512 + 256 + 256, [256, 256])
        self.fp3 = PointNetDecoder(128 + 128 + 256, [256, 256])
        self.fp2 = PointNetDecoder(32 + 64 + 256, [256, 128])
        self.fp1 = PointNetDecoder(128, [128, 128, 128])  # Original uses 128, not 128+3
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

        l1_xyz, l1_piece_id, l1_points = self.sa1(layer0_xyz, layer0_points, layer0_piece_id)
        l2_xyz, l2_piece_id, l2_points = self.sa2(l1_xyz, l1_points, l1_piece_id)
        l3_xyz, l3_piece_id, l3_points = self.sa3(l2_xyz, l2_points, l2_piece_id)
        l4_xyz, l4_piece_id, l4_points = self.sa4(l3_xyz, l3_points, l3_piece_id)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_piece_id, l4_piece_id, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_piece_id, l3_piece_id, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_piece_id, l2_piece_id, l1_points, l2_points)
        layer0_points = self.fp1(layer0_xyz, l1_xyz, layer0_piece_id, l1_piece_id, None, l1_points)

        feat = self.conv1(layer0_points)  # [1, feat_out, N_sum]
        feat = feat.permute(0, 2, 1).squeeze(0)  # [N_sum, feat_out]

        return feat
    
if __name__ == '__main__':
    xx = torch.randn(100, 3)
    bl = torch.tensor([20, 20, 35, 25], dtype=torch.long).reshape(4, 1)
    model_point = PointNetPTMSG(3, 128)
    feat_point = model_point(xx, bl)
    print(feat_point.shape)
