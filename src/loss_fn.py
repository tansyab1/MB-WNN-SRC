def loss_fn(sparsecodeLL, z_LL, z_LL_c, x_LL, x_LL_c, sparsecodeLH, z_LH, z_LH_c, x_LH, x_LH_c, sparsecodeHL, z_HL,
            z_HL_c, x_HL, x_HL_c, sparsecodeHH, z_HH, z_HH_c, x_HH, x_HH_c):
    recon_lossLL = 10 * F.mse_loss(x_LL_c, x_LL)
    express_lossLL =8 * F.mse_loss(z_LL, z_LL_c)
    reg_lossLL =   torch.norm(sparsecodeLL, p='fro')

    recon_lossLH = 10/1 * F.mse_loss(x_LH_c, x_LH)
    express_lossLH = 8/1 * F.mse_loss(z_LH, z_LH_c)
    reg_lossLH =   1/2 * torch.norm(sparsecodeLH, p='fro')

    recon_lossHL = 10/1 * F.mse_loss(x_HL_c, x_HL)
    express_lossHL =8/1 * F.mse_loss(z_HL, z_HL_c)
    reg_lossHL = 1/1 *  torch.norm(sparsecodeHL, p='fro')

    recon_lossHH = 10/1 * F.mse_loss(x_HH_c, x_HH)
    express_lossHH = 8/1 *F.mse_loss(z_HH, z_HH_c)
    reg_lossHH =   1/1* torch.norm(sparsecodeHH, p='fro')

    return recon_lossLL + recon_lossLH + recon_lossHL + recon_lossHH + express_lossLL + express_lossLH + express_lossHL + express_lossHH + reg_lossLL + reg_lossLH + reg_lossHL + reg_lossHH