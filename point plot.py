import scipy
import torch
import torch.nn.functional as F
import numpy as np
from ESRGAN_master.train1 import *
# from model.util import load_hrtf
# from model.dataset import get_sample_ratio
from config import Config
from ESRGAN_master.train1 import *
from ESRGAN_master.HRTF.evaluation.evaluation import *
import torch.nn.functional as F
from ESRGAN_master.HRTF.baselines.barycentric_interpolation import *
from ESRGAN_master.HRTF.baselines.hrtf_selection import *
from pathlib import Path
import importlib
from hrtfdata.transforms.hrirs import SphericalHarmonicsTransform
import matplotlib.pyplot as plt

def spectral_distortion_inner(input_spectrum, target_spectrum):
    numerator = target_spectrum
    denominator = input_spectrum
    return torch.mean((20 * np.log10(numerator / denominator)) ** 2)

def calc_lsd(ori_hrtf, recon_hrtf, domain):
    total_all_positions = 0
    total_positions = len(recon_hrtf)
    lsd_list = []
    for ori, gen in zip(ori_hrtf, recon_hrtf):
        if domain == 'magnitude_db':
            ori = 10 ** (ori/20)
            gen = 10 ** (gen/20)
        average_over_frequencies = spectral_distortion_inner(abs(gen), abs(ori))
        total_all_positions += np.sqrt(average_over_frequencies)
        lsd_list.append(np.sqrt(average_over_frequencies))
    sd_metric = total_all_positions / total_positions
    print('Log SD (across all positions): %s' % float(sd_metric))
    return np.array(lsd_list)

def replace_lsd(lsd_arr, upscale_factor):
    lsd_2d = lsd_arr.reshape(72,12)
    row_ratio, column_ratio = 36, 6
    for i in range(72 // row_ratio):
        for j in range(12 // column_ratio):
            lsd_2d[row_ratio*i, column_ratio*j] = 0
    return lsd_2d

# def plot_lsd(lsd_2d_1, lsd_2d_2, row_angles, column_angles, filename):
#     row_indices, col_indices = np.meshgrid(row_angles, column_angles)
#     x = row_indices.flatten()
#     y = col_indices.flatten()
#
#     values_1 = lsd_2d_1.T.flatten() # column for row angles, row for column angles
#     values_2 = lsd_2d_2.T.flatten()
#
#     min_value = min(np.min(values_1), np.min(values_2))
#     max_value = max(np.max(values_1), np.max(values_2))
#
#     plt.figure(figsize=(18, 10))
#     plt.subplot(2, 1, 1)
#     plt.scatter(x, y, c=values_1, cmap='OrRd', s=50, marker='o', edgecolor='black', vmin=min_value, vmax=max_value)
#     plt.colorbar(label='Average LSD Error')
#     plt.ylabel('Elevation (degree)')
#     plt.title('LSD for HRTF Selection-1')
#     plt.xticks([])  # Remove x-axis labels and ticks
#
#     plt.subplot(2, 1, 2)
#     plt.scatter(x, y, c=values_2, cmap='OrRd', s=50, marker='o', edgecolor='black', vmin=min_value, vmax=max_value)
#     plt.colorbar(label='Average LSD Error')
#     plt.xlabel('Azimuth (degree)')
#     plt.ylabel('Elevation (degree)')
#     plt.title('LSD for HRTF Selection-2')
#
#     plt.tight_layout()
#
#     plt.savefig(filename)
def plot_lsd(lsd_2d_1, lsd_2d_2, row_angles, column_angles, filename):
    row_indices, col_indices = np.meshgrid(row_angles, column_angles)
    x = row_indices.flatten()
    y = col_indices.flatten()

    values_1 = lsd_2d_1.T.flatten()  # column for row angles, row for column angles
    values_2 = lsd_2d_2.T.flatten()

    min_value = min(np.min(values_1), np.min(values_2))
    max_value = max(np.max(values_1), np.max(values_2))

    fig, axs = plt.subplots(2, 1, figsize=(18, 12))

    scatter_params = {
        'cmap': 'inferno',
        's': 60,
        'marker': 'o',
        'edgecolor': 'black',
        'vmin': min_value,
        'vmax': max_value
    }

    scatter_1 = axs[0].scatter(x, y, c=values_1, **scatter_params)
    fig.colorbar(scatter_1, ax=axs[0], label='Average LSD Error')
    axs[0].set_ylabel('Elevation (degree)', fontsize=14)
    axs[0].set_title('LSD for ESRGAN', fontsize=16)
    axs[0].grid(True, which="both", linestyle="--", linewidth=0.5)
    axs[0].set_xticks([])  # Remove x-axis labels and ticks

    scatter_2 = axs[1].scatter(x, y, c=values_2, **scatter_params)
    fig.colorbar(scatter_2, ax=axs[1], label='Average LSD Error')
    axs[1].set_xlabel('Azimuth (degree)', fontsize=14)
    axs[1].set_ylabel('Elevation (degree)', fontsize=14)
    axs[1].set_title('LSD for Barycentric Interpolation', fontsize=16)
    axs[1].grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.tight_layout(pad=2.0)
    plt.savefig(filename)
    plt.show()

# def plot_3d_lsd(lsd_2d, row_angles, column_angles, title, filename):
#     row_indices, col_indices = np.meshgrid(row_angles, column_angles)
#
#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(111, projection='3d')
#
#     ax.plot_surface(row_indices, col_indices, lsd_2d.T, cmap='OrRd', edgecolor='none', antialiased=True)
#
#     ax.set_xlabel('Azimuth (degree)')
#     ax.set_ylabel('Elevation (degree)')
#     ax.set_zlabel('Average LSD Error')
#     ax.set_title(title)
#
#     plt.savefig(filename)
#     # plt.show()

def plot_3d_lsd_combined(lsd_2d_1, lsd_2d_2, row_angles, column_angles, titles, filename):
    row_indices, col_indices = np.meshgrid(row_angles, column_angles)

    fig = plt.figure(figsize=(18, 12))

    # Plotting the first surface
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(row_indices, col_indices, lsd_2d_1.T, cmap='OrRd', edgecolor='none', antialiased=True)
    ax1.set_xlabel('Azimuth (degree)')
    ax1.set_ylabel('Elevation (degree)')
    ax1.set_zlabel('Average LSD Error')
    ax1.set_title(titles[0])

    # Plotting the second surface
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(row_indices, col_indices, lsd_2d_2.T, cmap='OrRd', edgecolor='none', antialiased=True)
    ax2.set_xlabel('Azimuth (degree)')
    ax2.set_ylabel('Elevation (degree)')
    ax2.set_zlabel('Average LSD Error')
    ax2.set_title(titles[1])

    plt.tight_layout()
    plt.savefig(filename)
    # plt.show()

def main():
    print("start visualize.py")
    # config = Config("ari-upscale-4", using_hpc=True)
    # config.upscale_factor = 216
    # data_dir = config.raw_hrtf_dir / config.dataset
    # imp = importlib.import_module('hrtfdata.full')
    # load_function = getattr(imp, config.dataset)
    # domain = config.domain
    # ds = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate,
    #                                                         'side': 'left', 'domain': domain}}, subject_ids='first')
    path = "C:/PycharmProjects/Upsample_GAN/ESRGAN_master/models/model"
    GPU = True  # Choose whether to use GPU
    if GPU:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(f'Using {device}')
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    config = Config("ari-upscale-4", using_hpc=False)
    data_dir = config.raw_hrtf_dir / config.dataset
    imp = importlib.import_module('hrtfdata.full')
    load_function = getattr(imp, config.dataset)
    ds = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate,
                                                         'side': 'left', 'domain': 'time'}})
    _, test_prefetcher = load_hrtf(config)
    # loader_train, loader_test = load_hrtf(config)
    row_angles = list(ds.row_angles)
    column_angles = list(ds.column_angles)
    num_row_angles = len(ds.row_angles)
    num_col_angles = len(ds.column_angles)
    num_radii = len(ds.radii)
    max_order = 21
    upscale_factor = config.upscale_factor
    degree = int(np.sqrt(num_row_angles*num_col_angles*num_radii/upscale_factor) - 1)

    # print("domain: ", domain, "upscale factor: ", upscale_factor)

    # ngpu = config.ngpu

    nbins = config.nbins_hrtf
    if config.merge_flag:
        nbins = config.nbins_hrtf * 2

    device = torch.device(config.device_name if (
        torch.cuda.is_available()) else "cpu")
    # model = AutoEncoder(nbins=nbins, in_order=degree, latent_dim=config.latent_dim, base_channels=256, num_features=512, out_oder=max_order)
    # print("Build VAE model successfully.")
    # model.load_state_dict(torch.load(f"{config.model_path}/Gen_216bn.pt", map_location=torch.device('cpu')))
    # print(f"Load VAE model weights `{os.path.abspath(config.model_path)}` successfully.")


    test_prefetcher.reset()
    batch_data = test_prefetcher.next()
    lr_coefficient = batch_data["lr_coefficient"].to(device=device, memory_format=torch.contiguous_format,
                                                        non_blocking=True, dtype=torch.float)
    hrtf = batch_data["hrir"]
    masks = batch_data["mask"]
    sample_id = 30

    # model.eval()
    # with torch.no_grad():
    #     recon = model(lr_coefficient)

    # original_mask = masks[0].numpy().astype(bool)
    # SHT = SphericalHarmonicsTransform(max_order, ds.row_angles, ds.column_angles, ds.radii, original_mask)
    # harmonics = torch.from_numpy(SHT.get_harmonics()).float().to(device)
    # recon_hrtf = harmonics @ recon[0].T
    # ori_hrtf = hrtf[0].reshape(nbins, -1).T
    # print("subject: ", sample_id)
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # file_name1 = f"/recon_SONICOM_{sample_id}.pickle"
    # with open("C:/PycharmProjects/Upsample_GAN/ESRGAN_master/plot" + file_name1, "rb") as f:
    #     recon_hrtf = pickle.load(f).permute(1, 2, 0, 3)  # r x w x h x nbins -> w x h x r x nbins
    # file_name2 = f"/ori_SONICOM_{sample_id}.pickle"
    # with open("C:/PycharmProjects/Upsample_GAN/ESRGAN_master/plot" + file_name2, "rb") as f:
    #     ori_hrtf = pickle.load(f).permute(1, 2, 0, 3)  # r x w x h x nbins -> w x h x r x nbins
    # lsd_arr = calc_lsd(ori_hrtf, recon_hrtf, domain='magnitude_db')
    # lsd_2d = replace_lsd(lsd_arr, upscale_factor)
    #
    # # for barycentric !!!!!!!!!!!!!!!!!
    # file_name2 = f"/ori_SONICOM_{sample_id}.pickle"
    # with open("C:/PycharmProjects/Upsample_GAN/ESRGAN_master/plot" + file_name2, "rb") as f:
    #     hr_hrtf = pickle.load(f).permute(1, 2, 0, 3)  # r x w x h x nbins -> w x h x r x nbins
    #
    # file_name = f"/bary_SONICOM_{sample_id}.pickle"
    # barycentric_data_folder = f'/barycentric_interpolated_data_{config.upscale_factor}'
    # barycentric_output_path = 'C:/PycharmProjects/Upsample_GAN/ESRGAN_master/plot'
    # with open(barycentric_output_path + file_name, "rb") as f:
    #     bary_hrtf = pickle.load(f)   # w x h x r x nbins
    #
    # hr_hrtf = hr_hrtf.reshape(-1, nbins)
    # bary_hrtf = bary_hrtf.reshape(-1, nbins)
    # lsd_arr_bary = calc_lsd(hr_hrtf, bary_hrtf, domain="magnitude")
    # lsd_2d_bary = replace_lsd(lsd_arr_bary, upscale_factor)
    # filename = f"lsd_select_{sample_id}.png"
    # plot_lsd(lsd_2d, lsd_2d_bary, row_angles, column_angles, filename)
    filename = f"lsd_select_{sample_id}.png"
    filename1 = "tiantian.pickle"
    with open("C:/PycharmProjects/Upsample_GAN/ESRGAN_master/plot/" + filename1, "rb") as f:
        lsd_2d, lsd_2d_bary = pickle.load(f)   # w x h x r x nbins
    plot_lsd(lsd_2d, lsd_2d_bary, row_angles, column_angles, filename)
    filename = f"lsd_select_3d_{sample_id}.png"
    titles = ['LSD for ESRGAN', 'LSD for Barycentric Interpolation']
    plot_3d_lsd_combined(lsd_2d, lsd_2d_bary, row_angles, column_angles, titles, filename)
    # ！！！！！！！！！！！！！！！！！！
    # for HRTF selection !!!!!!!!!!!!!!!!!
    # file_name = '/' + f"sample_{sample_id}.pickle"
    # with open(config.valid_gt_path + file_name, "rb") as f:
    #     hr_hrtf = pickle.load(f).permute(1, 2, 0, 3)  # r x w x h x nbins -> w x h x r x nbins

    # hr_hrtf = hr_hrtf.reshape(-1, nbins)
    #
    # with open(config.hrtf_selection_dir + "/maximum.pickle", "rb") as f:
    #     max_hrtf = pickle.load(f).reshape(-1, nbins)
    #
    # lsd_max_select = calc_lsd(hr_hrtf, max_hrtf, domain="magnitude")
    # lsd_max_2d = lsd_max_select.reshape(72,12)
    #
    # with open(config.hrtf_selection_dir + "/minimum.pickle", "rb") as f:
    #     min_hrtf = pickle.load(f).reshape(-1, nbins)
    #
    # lsd_min_select = calc_lsd(hr_hrtf, min_hrtf, domain="magnitude")
    # lsd_min_2d = lsd_min_select.reshape(72,12)
    #
    # filename = f"lsd_select_{sample_id}.png"
    # plot_lsd(lsd_min_2d, lsd_max_2d, row_angles, column_angles, filename)
    #
    # # filename = f"lsd_select_3d_{sample_id}.png"
    # # titles = ['LSD for Minimum Selection', 'LSD for Maximum Selection']
    # # plot_3d_lsd_combined(lsd_min_2d, lsd_max_2d, row_angles, column_angles, titles, filename)
    #
    # filename = f"lsd_select_3d_{sample_id}.png"
    # titles = ['LSD for HRTF Selection-1', 'LSD for HRTFSelection-2']
    # plot_3d_lsd_combined(lsd_min_2d, lsd_max_2d, row_angles, column_angles, titles, filename)


if __name__ == '__main__':
    main()

