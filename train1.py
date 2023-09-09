from torch.optim.lr_scheduler import *
import functools
from basicsr.utils.registry import ARCH_REGISTRY
from ESRGAN_master.HRTF.config import *
from ESRGAN_master.HRTF.hrtfdata.transforms.hrirs import SphericalHarmonicsTransform
from ESRGAN_master.HRTF.model.util import *
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from ESRGAN_master.train import *
from ESRGAN_master.HRTF.evaluation.evaluation import *
import torch.nn.functional as F
from torch import autograd
import gc
# import GPyOpt
# import GPy
import torch.optim as optim

path = "C:/PycharmProjects/Upsample_GAN/ESRGAN_master/models/model"
GPU = True  # Choose whether to use GPU
if GPU:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(f'Using {device}')
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
config = Config("ari-upscale-4", using_hpc=False)
data_dir = config.raw_hrtf_dir / config.dataset
imp = importlib.import_module('hrtfdata.full')
load_function = getattr(imp, config.dataset)


# #### Data loading

# In[4]:
# def make_layer(block, n_layers):
#     layers = []
#     for _ in range(n_layers):
#         layers.append(block())
#     return nn.Sequential(*layers)
def make_layer(block, n_layers, **kwargs):
    layers = []
    for _ in range(n_layers):
        layers.append(block(**kwargs))
    return nn.Sequential(*layers)

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv1d(in_dim, in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_dim, in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, W = x.size()
        query = self.query_conv(x).view(B, -1, W).permute(0, 2, 1)
        key = self.key_conv(x).view(B, -1, W)
        attention = torch.bmm(query, key)
        attention = F.tanh(attention) # dim=-1
        # print(f"attention{attention.shape}")
        value = self.value_conv(x).view(B, -1, W)
        # print(value.shape)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, W)
        out = self.gamma*out + x
        return out


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv1d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv1d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv1d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv1d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv1d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    # def forward(self, x):
    #     x1 = self.lrelu(self.conv1(x))
    #     x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
    #     x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
    #     x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
    #     x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
    #     return x5 * 0.2 + x
    def forward(self, x):
        x1 = F.tanh(self.conv1(x))
        x2 = F.tanh(self.conv2(torch.cat((x, x1), 1)))
        x3 = F.tanh(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = F.tanh(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)
        # self.RDB4 = ResidualDenseBlock_5C(nf, gc)
        # self.RDB5 = ResidualDenseBlock_5C(nf, gc)
        self.attention = SelfAttention(nf)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        # out = self.RDB4(out)
        # out = self.RDB5(out)
        out = self.attention(out)  # Apply self-attention here
        return out * 0.2 + x

class Upsample(nn.Module):
    def __init__(self, scale_factor):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')

class Downsample(nn.Module):
    def __init__(self, output_size):
        super(Downsample, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return x[:, :, :self.output_size]
        # return F.adaptive_avg_pool1d(x, self.output_size)

class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.dropout = nn.Dropout(0.5)

        self.conv_first = nn.Conv1d(in_nc, nf, 3, 1, 1, bias=True)
        # self.conv_first = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.RRDB_trunk1 = make_layer(RRDB_block_f, nb)
        self.RRDB_trunk2 = make_layer(RRDB_block_f, nb)
        self.RRDB_trunk3 = make_layer(RRDB_block_f, nb)
        self.RRDB_trunk4 = make_layer(RRDB_block_f, nb)
        self.RRDB_trunk5 = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv1d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv1d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv1d(nf, nf, 3, 1, 1, bias=True)
        self.upconv3 = nn.Conv1d(nf, nf, 3, 1, 1, bias=True)
        self.upconv4 = nn.Conv1d(nf, nf, 3, 1, 1, bias=True)
        # self.upconv5 = nn.Conv1d(nf, nf, 3, 1, 1, bias=True)
        # self.upconv6 = nn.Conv1d(nf, nf, 3, 1, 1, bias=True)
        # self.upconv7 = nn.Conv1d(nf, nf, 3, 1, 1, bias=True)


        self.HRconv = nn.Conv1d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv1d(nf, nf, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Upsampling and Downsampling layers
        self.up = Upsample(scale_factor=3)  # 784*3 = 2352
        self.down = Downsample(output_size=484)  # downsample to 841

        self.conv = nn.Conv1d(nf, nf, 3, 1, 1)

    def forward(self, x):
        fea = self.conv_first(x)
        # print("f1: ", fea.shape)
        trunk = self.trunk_conv(self.RRDB_trunk1(fea))
        fea = fea + trunk

        fea = F.tanh(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        trunk = self.trunk_conv(self.RRDB_trunk2(fea))
        fea = fea + trunk
        fea = nn.Dropout(0.5)(fea)

        fea = F.tanh(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        trunk = self.trunk_conv(self.RRDB_trunk3(fea))
        fea = fea + trunk
        fea = nn.Dropout(0.5)(fea)

        fea = F.tanh(self.upconv3(F.interpolate(fea, scale_factor=2, mode='nearest')))
        trunk = self.trunk_conv(self.RRDB_trunk4(fea))
        fea = fea + trunk
        fea = nn.Dropout(0.5)(fea)

        fea = F.tanh(self.upconv4(F.interpolate(fea, scale_factor=2, mode='nearest')))
        trunk = self.trunk_conv(self.RRDB_trunk5(fea))
        fea = fea + trunk
        fea = nn.Dropout(0.5)(fea)
        # print("f6: ", fea.shape)
        # fea = nn.Dropout(0.5)(fea)
        # fea = F.tanh(self.upconv5(F.interpolate(fea, scale_factor=2, mode='nearest')))
        # print("f7: ", fea.shape)
        # fea = nn.Dropout(0.5)(fea)
        # fea = F.tanh(self.upconv6(F.interpolate(fea, scale_factor=2, mode='nearest')))
        # print("f8: ", fea.shape)
        # fea = nn.Dropout(0.5)(fea)
        # fea = F.tanh(self.upconv7(F.interpolate(fea, scale_factor=2, mode='nearest')))
        # fea = nn.Dropout(0.5)(fea)
        out = self.conv_last(fea)
        # print("f9: ", out.shape)
        # Apply Upsampling and Downsampling
        out = self.up(out)
        out = self.down(out)
        out = self.conv(out)
        # print(out.shape)
        return out

@ARCH_REGISTRY.register()
class VGGStyleDiscriminator1D_v2(nn.Module):
    def __init__(self, num_in_ch, num_feat, input_size=484):
        super(VGGStyleDiscriminator1D_v2, self).__init__()
        self.input_size = input_size
        self.dropout = nn.Dropout(0.5)

        self.conv0_0 = nn.Conv1d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv1d(num_feat, num_feat, 3, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm1d(num_feat, affine=True)

        self.conv1_0 = nn.Conv1d(num_feat, num_feat * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm1d(num_feat * 2, affine=True)
        self.conv1_1 = nn.Conv1d(num_feat * 2, num_feat * 2, 3, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm1d(num_feat * 2, affine=True)
        # 1664, 100
        self.conv2_0 = nn.Conv1d(num_feat * 2, num_feat * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm1d(num_feat * 4, affine=True)
        self.conv2_1 = nn.Conv1d(num_feat * 4, num_feat * 4, 3, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm1d(num_feat * 4, affine=True)

        self.conv3_0 = nn.Conv1d(num_feat * 4, num_feat * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm1d(num_feat * 8, affine=True)
        self.conv3_1 = nn.Conv1d(num_feat * 8, num_feat * 8, 3, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm1d(num_feat * 8, affine=True)

        self.conv4_0 = nn.Conv1d(num_feat * 8, num_feat * 10, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm1d(num_feat * 10, affine=True)
        self.conv4_1 = nn.Conv1d(num_feat * 10, num_feat * 10, 3, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm1d(num_feat * 10, affine=True)

        self.linear1 = nn.Linear(10240, 512)  # Adjusted input dimension  16384  [10240, 100]
        self.linear2 = nn.Linear(512, 100)
        self.linear3 = nn.Linear(100, 1)

            # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)


    # def forward(self, x):
    #     assert x.size(2) == self.input_size, (f'Input size must be identical to input_size, but received {x.size()}.')
    #
    #     feat = self.lrelu(self.conv0_0(x))
    #     feat = self.lrelu(self.bn0_1(self.conv0_1(feat)))
    #     feat = self.dropout(feat)
    #
    #     feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
    #     feat = self.dropout(feat)
    #     feat = self.lrelu(self.bn1_1(self.conv1_1(feat)))
    #     feat = self.dropout(feat)
    #
    #     # spatial size: (4, 4)
    #     feat = feat.view(feat.size(0), -1)
    #     feat = self.lrelu(self.linear1(feat))
    #     feat = self.dropout(feat)
    #     out = self.linear2(feat)
    #     return out
    def forward(self, x):
        # assert x.size(2) == self.input_size, (f'Input size must be identical to input_size, but received {x.size()}.')

        feat = self.lrelu(self.conv0_0(x))
        # print("f1: ", feat.shape)
        feat = self.lrelu(self.bn0_1(self.conv0_1(feat)))
        # print("f2: ", feat.shape)
        # feat = nn.AlphaDropout(0.5)(feat)
        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
        # print("f3: ", feat.shape)
        # feat = nn.AlphaDropout(0.5)(feat)
        feat = self.lrelu(self.bn1_1(self.conv1_1(feat)))
        # print("f4: ", feat.shape)
        # feat = nn.AlphaDropout(0.5)(feat)
        # spatial size: (4, 4)
        feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
        # print("f5: ", feat.shape)
        # print("f3: ", feat.shape)
        # feat = nn.AlphaDropout(0.5)(feat)
        feat = self.lrelu(self.bn2_1(self.conv2_1(feat)))
        # print("f6: ", feat.shape)
        # print("f4: ", feat.shape)
        # feat = nn.AlphaDropout(0.5)(feat)
        feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
        # print("f5: ", feat.shape)
        # print("f3: ", feat.shape)
        # feat = nn.AlphaDropout(0.5)(feat)
        feat = self.lrelu(self.bn3_1(self.conv3_1(feat)))
        # print("f6: ", feat.shape)
        # print("f4: ", feat.shape)
        # feat = nn.AlphaDropout(0.5)(feat)

        feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
        # print("f3: ", feat.shape)
        # feat = nn.AlphaDropout(0.5)(feat)
        feat = self.lrelu(self.bn4_1(self.conv4_1(feat)))
        # print("f4: ", feat.shape)
        # feat = nn.AlphaDropout(0.5)(feat)

        feat = feat.view(feat.size(0), -1)
        feat = self.lrelu(self.linear1(feat))    # lrelu
        # feat = self.dropout(feat)
        feat = self.lrelu(self.linear2(feat))
        # feat = self.dropout(feat)
        out = self.linear3(feat)
        out = F.sigmoid(out)
        return out

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)
def weights_init(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.fill_(0)


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1))).to(real_samples.device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size()).to(real_samples.device)

    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def main():
    # Apply data loader
    config = Config("ari-upscale-4", using_hpc=False)
    imp = importlib.import_module('hrtfdata.full')
    load_function = getattr(imp, config.dataset)
    ds = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate,
                                                         'side': 'left', 'domain': 'time'}})
    # Preprocessing
    ########################################################################################
    train_size = int(len(set(ds.subject_ids)) * config.train_samples_ratio)
    train_sample = np.random.choice(list(set(ds.subject_ids)), train_size, replace=False)
    val_sample = list(set(ds.subject_ids) - set(train_sample))
    id_file_dir = config.train_val_id_dir
    if not os.path.exists(id_file_dir):
        os.makedirs(id_file_dir)
    id_filename = id_file_dir + '/train_val_id.pickle'
    with open(id_filename, "wb") as file:
        pickle.dump((train_sample, val_sample), file)
    #########################################################################################
    loader_train, loader_test = load_hrtf(config)

    num_epochs = 150
    use_weights_init = True

    model_G = RRDBNet(256, 128, 256, 23, gc=32).to(device)
    # model_G = model_G.double()
    model_G = model_G.float()
    if use_weights_init:
        model_G.apply(weights_init)
    params_G = sum(p.numel() for p in model_G.parameters() if p.requires_grad)
    print("Total number of parameters in Generator is: {}".format(params_G))
    print(model_G)
    print('\n')
    # [256, 64]
    model_D = VGGStyleDiscriminator1D_v2(256, 64).to(device)
    # model_D = model_D.double()
    # double
    model_D = model_D.float()
    if use_weights_init:
        model_D.apply(weights_init)
    params_D = sum(p.numel() for p in model_D.parameters() if p.requires_grad)
    print("Total number of parameters in Discriminator is: {}".format(params_D))
    print(model_D)
    print('\n')

    print("Total number of parameters is: {}".format(params_G + params_D))

    def loss_function(out, label):
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(out, label)
        return loss
    def G_loss_function(sr, hr_coefficient):
        cos_similarity_criterion = nn.CosineSimilarity(dim=2)
        sh_cos_loss = 1 - cos_similarity_criterion(sr, hr_coefficient).mean()
        return sh_cos_loss
    Disc_lr = 2e-4 # 2e-4
    beta1 = 0.5
    Gene_lr = 1e-4 # 1e-4
    optimizerD = torch.optim.Adam(model_D.parameters(), lr=Disc_lr, betas=(beta1, 0.999))
    optimizerG = torch.optim.Adam(model_G.parameters(), lr=Gene_lr, betas=(beta1, 0.999))

    # Additional input variables should be defined here
    False_img_label = 0
    # Mode Collapse can be avoid by initalise Real_label to 0.9 instead of 1
    True_img_label = 0.9

    # Start training loop
    train_losses_G = []
    train_losses_D = []
    content_losses = []
    adv_losses_G = []
    error_D = []
    generator_gradient_norms = []
    discriminator_gradient_norms = []
    schedulerD = StepLR(optimizerD, step_size=20, gamma=0.93)

    for epoch in range(num_epochs):
        train_loss_D = 0
        train_loss_G = 0
        total_content_loss = 0
        train_adv_G = 0
        lambda_gp = 10  # Gradient penalty lambda hyperparameter
        loader_train.reset()
        batch_data = loader_train.next()
        print("Epoch:", epoch)
        counter = 0
        while batch_data is not None:
            model_D.zero_grad()
            lr_coefficient = batch_data["lr_coefficient"].to(device=device, memory_format=torch.contiguous_format,
                                                             non_blocking=True)
            hr_coefficient = batch_data["hr_coefficient"].to(device=device, memory_format=torch.contiguous_format,
                                                             non_blocking=True)
            print(lr_coefficient.shape)
            print(hr_coefficient.shape)
            # hr_coefficient = hr_coefficient.view([1, 256, 2116])
            # lr_coefficient = batch_data["lr_coefficient"].to(device=device)
            # lr_coefficient = lr_coefficient.float()
            # hr_coefficient = batch_data["hr_coefficient"].to(device=device)
            # hr_coefficient = hr_coefficient.float()
            hrir = batch_data["hrir"].to(device=device, memory_format=torch.contiguous_format,
                                         non_blocking=True)
            masks = batch_data["mask"]
            lr_coefficient = lr_coefficient.float()
            # print(lr_coefficient.size())
            hr_coefficient = hr_coefficient.float()
            hrir = hrir.float()
            size_batch = lr_coefficient.size(0)
            # img_label = torch.full((size_batch,), True_img_label, device=device, dtype=torch.float)
            img_label = torch.full((size_batch,), True_img_label, device=device)
            img_label = img_label.reshape(-1, 1)
            real_D_output = model_D(hr_coefficient)
            real_D_error = loss_function(real_D_output, img_label)
            # real_D_error.backward()
            real_D_error.backward(retain_graph=True)
            # Output true mean discriminator result
            D_x = real_D_output.mean().item()

            # Start to train discriminator with fake image
            # [1, 128, 49]
            img_fake = model_G(lr_coefficient)
            # print(img_fake.shape)
            lr0 = lr_coefficient[0].T
            sr0 = img_fake[0].T
            hr0 = hr_coefficient[0].T
            print(f"lr: {lr0.shape}, {lr0[0, :20]}\n")
            print(f"sr: {sr0.shape}, {sr0[0, :20]}\n")
            print(f"hr: {hr0.shape}, {hr0[0, :20]}\n")
            # print(img_fake.size())
            img_label.fill_(False_img_label)
            real_D_output = model_D(img_fake.detach())
            fake_D_error = loss_function(real_D_output, img_label)

            # fake_D_error.backward()
            fake_D_error.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model_D.parameters(), max_norm=1.0)
            discriminator_gradient_norm = torch.sqrt(
                sum(p.grad.data.norm() ** 2 for p in model_D.parameters() if p.grad is not None)
            )
            discriminator_gradient_norms.append(discriminator_gradient_norm.item())
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(model_D, hr_coefficient.data, img_fake.data)
            penalty_loss = lambda_gp * gradient_penalty
            penalty_loss.backward()

            errD = real_D_error + fake_D_error + penalty_loss
            print("errD: ", errD.item())
            print("D real: ", real_D_error.item())
            print("D fake: ", fake_D_error.item())
            print("penality: ", penalty_loss.item())
            train_loss_D = train_loss_D + errD.item()
            # D_G_z1 = real_D_output.mean().item()
            # errD = real_D_error + fake_D_error
            # train_loss_D = train_loss_D + errD.item()
            optimizerD.step()
            # del lr_coefficient
            # del hr_coefficient
            # del hrir
            # torch.cuda.empty_cache()
            # Start to maximizing log(D(G(z))) to update G network
            model_G.zero_grad()
            content_criterion = sd_ild_loss
            recon_coef_list = []
            ds = load_function(data_dir, feature_spec={
                'hrirs': {'samplerate': config.hrir_samplerate, 'side': 'left', 'domain': 'magnitude_db'}})
            num_row_angles = len(ds.row_angles)
            num_col_angles = len(ds.column_angles)
            num_radii = len(ds.radii)
            # HRTF Frequency(one ear 128, if concate two ears, it should be 256)
            nbins = config.nbins_hrtf
            # mean and std for ILD and SD, which are used for normalization
            # computed based on average ILD and SD for training data, when comparing each individual
            # to every other individual in the training data
            sd_mean = 7.387559253346883
            sd_std = 0.577364154400081
            ild_mean = 3.6508303231127868
            ild_std = 0.5261339271318863
            if config.merge_flag:
                nbins = config.nbins_hrtf * 2
            for i in range(masks.size(0)):
                SHT = SphericalHarmonicsTransform(21, ds.row_angles, ds.column_angles, ds.radii,
                                                  masks[i].detach().cpu().numpy().astype(bool))
                # harmonics = torch.from_numpy(SHT.get_harmonics()).float().to(device)
                harmonics = torch.from_numpy(SHT.get_harmonics()).float().to(device)
                # recon_hrir = SHT.inverse(img_fake[i].T.detach().cpu().numpy())  # Compute the inverse
                # recon_hrir = harmonics @ img_fake[i].T.detach().cpu().numpy()
                recon_hrir = (harmonics @ img_fake[i].T)
                # recon_hrir = torch.from_numpy(recon_hrir).type(torch.double).to(device)
                # recon_hrir = recon_hrir.type(torch.float32)
                # recon_hrir_tensor = torch.from_numpy(recon_hrir.T).reshape(nbins, num_radii, num_row_angles, num_col_angles)
                recon_hrir_tensor = recon_hrir.T.reshape(nbins, num_radii, num_row_angles,
                                                                       num_col_angles)
                # recon_hrir_tensor = (harmonics @ recon_hrir.T).reshape(num_row_angles, num_col_angles,
                #                                                              num_radii, nbins)
                # recon_hrir_tensor = F.relu(recon_hrir_tensor.permute(3, 2, 0, 1))
                # recon_hrir_tensor = F.softplus(
                #     recon_hrir_tensor.reshape(-1, nbins, num_radii, num_row_angles, num_col_angles))
                recon_coef_list.append(recon_hrir_tensor)
            counter += 1
            recons = torch.stack(recon_coef_list).to(device)
            plot_recon = torch.permute(recons[0], (2, 3, 1, 0)).detach().cpu()
            plot_hrtf = torch.permute(hrir[0], (2, 3, 1, 0)).detach().cpu()

            x = plot_recon[12, 2, 0, :]
            y = plot_hrtf[12, 2, 0, :]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.plot(x)
            ax1.set_title('recon')
            ax2.plot(y)
            ax2.set_title('original')
            plt.savefig(f"output_{epoch}.png")
            plt.close()
            unweighted_content_loss = content_criterion(config, recons, hrir, sd_mean, sd_std, ild_mean, ild_std)
            content_loss = config.content_weight * unweighted_content_loss
            img_label.fill_(True_img_label)
            img_fake_result = model_D(img_fake)
            adversarialG = loss_function(img_fake_result, img_label)
            coefficient_loss = G_loss_function(img_fake, hr_coefficient)
            train_adv_G += adversarialG.item()
            errG = adversarialG + content_loss + coefficient_loss
            errG.backward()
            torch.nn.utils.clip_grad_norm_(model_G.parameters(), max_norm=1.0)
            generator_gradient_norm = torch.sqrt(
                sum(p.grad.data.norm() ** 2 for p in model_G.parameters() if p.grad is not None)
            )
            generator_gradient_norms.append(generator_gradient_norm.item())
            D_G_z2 = img_fake_result.mean().item()
            train_loss_G = train_loss_G + errG.item()
            print("errG: ", errG)
            total_content_loss += adversarialG.item()
            print("content loss: ", content_loss)
            print("adversarial G: ", adversarialG)
            optimizerG.step()
            # del recons
            # torch.cuda.empty_cache()

            #######################################################################
            #                       ** END OF YOUR CODE **
            #######################################################################
            # Logging
            if counter % 50 == 0:
                print(f"Batch:{counter}/{len(loader_train)}")
            # if counter == 162:
                # torch.save(model_G.state_dict(), f'{path}/ GAN_G_model.pth')
                # torch.save(model_D.state_dict(), f'{path}/ GAN_D_model.pth')
            i += 1
            # del batch_data
            batch_data = loader_train.next()
        schedulerD.step()
        train_losses_D.append(train_loss_D / len(loader_train))
        train_losses_G.append(train_loss_G / len(loader_train))
        content_losses.append(total_content_loss / len(loader_train))
        adv_losses_G.append(train_adv_G / len(loader_train))
        error_D.append(errD / len(loader_train))
        batch_data = loader_train.next()
        print("Finish")
        torch.cuda.empty_cache()

        # Delete large tensors that aren't needed anymore to free up memory
        del batch_data
        del lr_coefficient
        del hr_coefficient
        del hrir
        del img_fake
        del recons
        # Explicitly run the Python garbage collector
        gc.collect()
    torch.save(model_G.state_dict(), f'{path}/GAN_G_model_{num_epochs}.pth')
    torch.save(model_D.state_dict(), f'{path}/GAN_D_model_{num_epochs}.pth')

    # Plot loss curves
    figure, axis = plt.subplots(1, 2, figsize=(16, 5))
    figure.suptitle(f'the losses curves for the discriminator D and the generator G as the training progresses')

    axis[0].set_xlabel('Number of Epochs')
    axis[0].set_ylabel('Model Loss')
    axis[0].plot(train_losses_G, 'b', label='Generator')
    leg = axis[0].legend(loc='upper right')

    axis[1].set_xlabel('Number of Epochs')
    axis[1].set_ylabel('Model Loss')
    axis[1].plot(train_losses_D, 'r', label='Discriminator')
    leg = axis[1].legend(loc='upper right')
    # Save the figure before calling plt.show()
    plt.savefig(f'loss_curve_{num_epochs}.png', dpi=300)
    plt.show()

    # Plot content_loss curve
    plt.figure(figsize=(8, 5))
    plt.title('The content losses curve for the Generator as the training progresses')

    plt.xlabel('Number of Epochs')
    plt.ylabel('Model Loss')
    plt.plot(content_losses, 'b', label='Generator')
    # plt.plot(content_loss, 'b', label='Generator')
    plt.legend(loc='upper right')
    # Save the figure before calling plt.show()
    plt.savefig(f'content_loss_curve_{num_epochs}.png', dpi=300)
    plt.show()

    # Plot adv_G_loss curve
    plt.figure(figsize=(8, 5))
    plt.title('The adversarial losses curve for the Generator as the training progresses')

    plt.xlabel('Number of Epochs')
    plt.ylabel('Adversarial Loss')
    plt.plot(adv_losses_G, 'b', label='Generator')
    # plt.plot(content_loss, 'b', label='Generator')
    plt.legend(loc='upper right')
    # Save the figure before calling plt.show()
    plt.savefig(f'adversarial_loss_curve_{num_epochs}.png', dpi=300)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.title('The error D losses curve for the Discriminator as the training progresses')

    plt.xlabel('Number of Epochs')
    plt.ylabel('Model Loss')
    plt.plot([tensor.detach().cpu().numpy() for tensor in error_D], 'b', label='Discriminator')
    plt.legend(loc='upper right')
    # Save the figure before calling plt.show()
    plt.savefig(f'errorD_curve_{num_epochs}.png', dpi=300)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.title('Gradient Norms')
    plt.plot(generator_gradient_norms, label='Generator')
    plt.plot(discriminator_gradient_norms, label='Discriminator')
    plt.xlabel('Epochs')
    plt.ylabel('Norm')
    plt.legend()
    plt.savefig(f'Gradient_Norms_curve_{num_epochs}.png', dpi=300)
    plt.show()


#  # Secondly, try to use Bayesian optimization search to perform hyperparameter tuning
# # Initialise some dropout rate for tuning
# rate_dropout = [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6]
# # Initialise my weight method
# method_for_weight = [nn.init.kaiming_uniform_,
#           nn.init.kaiming_normal_,
#           nn.init.xavier_uniform_,
#           nn.init.xavier_normal_]
#
#
# # Choose which model to run test, Resnet in this cw
# # current_model = MyResNet(weight_method=method_for_weight[0], dropout=rate_dropout[2])
# current_model = RRDBNet(256, 128, 256, 23, gc=32).to(device).float()
# current_model = current_model.float()
# current_optimizer = optim.Adam(current_model.parameters(), lr=0.0005, weight_decay=0.0000841)
# # Calculate amount of parameters
# total_parameters = sum(p.numel() for p in current_model.parameters() if p.requires_grad)
# print("The amount of model parameters is: {}".format(total_parameters))
# # Start train
# train_part(current_model, current_optimizer, epochs = 10)
#
# # Show final test accuracy
# check_accuracy(loader_val, current_model, analysis=True)
#
# # Finally, save current model
# torch.save(current_model.state_dict(), 'model.pt')
# method_for_weight = [nn.init.kaiming_uniform_,
#                      nn.init.kaiming_normal_,
#                      nn.init.xavier_uniform_,
#                      nn.init.xavier_normal_]
#
#
# def Bayesian_Optimization_Search(hyperparameter):
#     # Choose resnet to do tuning process
#     current_model = MyResNet(weight_method=method_for_weight[0], dropout=rate_dropout[2])
#     # Squeeze all hyperparameters
#     current_hyperparameter = hyperparameter.squeeze()
#     # Choose optimizer of model
#     current_optimizer = optim.Adam(current_model.parameters(), lr=current_hyperparameter[0], weight_decay=current_hyperparameter[1])
#     # Setup randomly train batch
#     rand_num = int(torch.rand(1) * 40000)
#     print("Randomly selected train batch: %d-%d Learning, Rate = %.6f, Weight Decay = %.6f" % (
#     rand_num, rand_num + 1000, current_hyperparameter[0], current_hyperparameter[1]))
#     train_part(current_model, current_optimizer, epochs=10)
#     return check_accuracy(loader_val, current_model)
#
# # Initialise keys
# keys = [{'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-5, 1e-2)},
#           {'name': 'weight_decay', 'type': 'continuous', 'domain': (1e-5, 1e-2)}]
# # Start optimise parameters by using bayesian optimization search
# test_Bayesian_Optimization = GPyOpt.methods.BayesianOptimization(f=Bayesian_Optimization_Search,
#                                               domain=keys,
#                                               model_type='GP',
#                                               acquisition_type='EI',
#                                               maximize=True)
#
#
# test_Bayesian_Optimization.run_optimization(20, 60, 10e-6)
# # Plot result to better comparison improvement
# test_Bayesian_Optimization.plot_convergence()
# lr, wd = test_Bayesian_Optimization.x_opt
# print("Obtain! Current best Hyperparameter with lr = %.7f, Weight Decay = %.7f, obtained accuracy: %.2f" % (
# lr, wd, test_Bayesian_Optimization.fx_opt))

if __name__ == '__main__':
    main()




