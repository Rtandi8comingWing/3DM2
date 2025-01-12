class Mamba3DBlock(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.SiLU,
                 norm_layer=nn.LayerNorm,
                 k_group_size=8,
                 alpha=100,
                 beta=1000,
                 num_group=128,
                 num_heads=6,
                 bimamba_type="v2",
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        self.num_group = num_group
        self.k_group_size = k_group_size

        self.num_heads = num_heads

        # self.lfa = LNPBlock(lga_out_dim=dim * 2,
        #                     k_group_size=self.k_group_size,
        #                     alpha=alpha,
        #                     beta=beta,
        #                     mlp_in_dim=dim * 2,
        #                     mlp_out_dim=dim,
        #                     num_group=self.num_group,
        #                     act_layer=act_layer,
        #                     drop_path=drop_path,
        #                     # num_heads=self.num_heads, # uncomment this line if use attention
        #                     norm_layer=norm_layer,
        #                     )

        self.model = Mamba(dim, bimamba_type=bimamba_type)

    # def shuffle_x(self, x, shuffle_idx):
    #     pos = x[:, None, 0, :]
    #     feat = x[:, 1:, :]
    #     shuffle_feat = feat[:, shuffle_idx, :]
    #     x = torch.cat([pos, shuffle_feat], dim=1)
    #     return x
    #
    # def mamba_shuffle(self, x):
    #     G = x.shape[1] - 1  #
    #     shuffle_idx = torch.randperm(G)
    #     # shuffle_idx = torch.randperm(int(0.4*self.num_group+1)) # 1-mask
    #     x = self.shuffle_x(x, shuffle_idx)  # shuffle
    #
    #     x = self.mixer(self.norm2(x))  # layernorm->mamba
    #
    #     x = self.shuffle_x(x, shuffle_idx)  # un-shuffle
    #     return x

    def forward(self, center, x):
        # x + norm(x)->lfa(x)->dropout
        # x = x + self.drop_path(self.lfa(center, self.norm1(x)))  # x: 32 129 384. center: 32 128 3

        # x + norm(x)->mamba(x)->dropout
        # x = x + self.drop_path(self.mamba_shuffle(x))
        x = x + self.drop_path(self.mixer(self.norm2(x)))

        return x