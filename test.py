from torch.optim.lr_scheduler import *
import functools
from basicsr.utils.registry import ARCH_REGISTRY
from ESRGAN_master.HRTF.config import *
from ESRGAN_master.HRTF.hrtfdata.transforms.hrirs import SphericalHarmonicsTransform
from ESRGAN_master.HRTF.model.util import *
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from ESRGAN_master.HRTF.evaluation import *
from ESRGAN_master.train1 import *
from ESRGAN_master.HRTF.evaluation.evaluation import *
import torch.nn.functional as F
from ESRGAN_master.HRTF.baselines.barycentric_interpolation import *
from ESRGAN_master.HRTF.baselines.hrtf_selection import *

def main():
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
    loader_train, loader_test = load_hrtf(config)

    counter = 0

    plot_flag = True
    for batch_data in loader_test.data:
        lr_coefficient = batch_data["lr_coefficient"].to(device=device).double()
        hr_coefficient = batch_data["hr_coefficient"].to(device=device, memory_format=torch.contiguous_format,
                                                         non_blocking=True)
        hrtf = batch_data["hrir"]
        masks = batch_data["mask"].to(device=device, memory_format=torch.contiguous_format,
                                      non_blocking=True, dtype=torch.double)
        model_G = RRDBNet(256, 128, 256, 23, gc=32).to(device)
        model_G = model_G.double()
        model_G.load_state_dict(torch.load("C:/PycharmProjects/Upsample_GAN/result/upsacle_216_attention_enhanced/enhanced/GAN_G_model_150.pth"))
        # model_G.load_state_dict(torch.load("C:/PycharmProjects/Upsample_GAN_HPC/result/up_scale_216/GAN_G_model_100.pth"))
        # C:/PycharmProjects/Upsample_GAN/ESRGAN_master/models/model/GAN_G_model.pth
        with torch.no_grad():
            generated = model_G(lr_coefficient)
            # print(generated-hr_coefficient)
            SHT = SphericalHarmonicsTransform(21, ds.row_angles, ds.column_angles, ds.radii,
                                              masks[0].detach().cpu().numpy().astype(bool))
            harmonics = torch.from_numpy(SHT.get_harmonics()).double().to(device)
            recon_hrir = (harmonics @ generated[0].T).detach().cpu().numpy()
            recon_hrir = torch.from_numpy(recon_hrir).type(torch.double)

            # Rest of the code remains the same
            num_row_angles = len(ds.row_angles)
            num_col_angles = len(ds.column_angles)
            num_radii = len(ds.radii)
            nbins = config.nbins_hrtf
            if config.merge_flag:
                nbins = config.nbins_hrtf * 2
            # print(recon_hrir_tensor.size())
            recon_hrir_tensor = recon_hrir.transpose(0, 1).reshape(-1, nbins, num_radii, num_row_angles,
                                                                   num_col_angles)
            # recon_hrir_tensor = F.relu(recon_hrir_tensor.reshape(-1, nbins, num_radii, num_row_angles, num_col_angles))
            if plot_flag:
                plot_recon = torch.permute(recon_hrir_tensor[0], (2, 3, 1, 0)).detach().cpu()
                plot_hrtf = torch.permute(hrtf[0], (2, 3, 1, 0)).detach().cpu()

                x = plot_recon[12, 2, 0, :]
                y = plot_hrtf[12, 2, 0, :]
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                ax1.plot(x)
                ax1.set_title('recon')
                ax2.plot(y)
                ax2.set_title('original')
                plt.savefig("output.png")
                plot_flag = False


            recon_hrir_tensor = torch.permute(recon_hrir_tensor[0], (2, 3, 1, 0)).detach().cpu()  # w * h * r * nbins
            hrtf = torch.permute(hrtf[0], (1, 2, 3, 0)).detach().cpu()
            # print(recon_hrir_tensor)
            # print(hrtf)
            file_name = f"/sample_{counter}.pickle"
            with open("C:/PycharmProjects/Upsample_GAN/runs-hpc/ari-upscale-4/valid" + file_name, 'wb') as file:
                pickle.dump(recon_hrir_tensor, file)
            with open("C:/PycharmProjects/Upsample_GAN/runs-hpc/ari-upscale-4/valid_gt" + file_name, 'wb') as file:
                pickle.dump(hrtf, file)
            counter += 1
    #######################################################################################################################
    # # Evaluation
    # _, test_prefetcher = load_hrtf(config)
    # print("Loaded all datasets successfully.")
    #
    # run_lsd_evaluation(config, config.valid_path)
    # run_localisation_evaluation(config, config.valid_path)
    #######################################################################################################################
    # barycentric interpolation
    barycentric_output_path = 'C:/PycharmProjects/Upsample_GAN/ESRGAN_master/HRTF/baselines/barycentric_interpolated_data'
    # barycentric_output_path = config.barycentric_hrtf_dir + barycentric_data_folder
    # run_barycentric_interpolation(config, barycentric_output_path)
    # print("!!!!!!!!!!!!!!!!!!my interpolation!!!!!!!!!!!!!!!!!!!!!!!!")
    # debug_barycentric(config, barycentric_output_path)
    sphere_coords = my_barycentric_interpolation(config, barycentric_output_path)
    # sphere_coords = my_barycentric_interpolation(config, barycentric_output_path)
    if config.gen_sofa_flag:
        row_angles = list(set([x[1] for x in sphere_coords]))
        column_angles = list(set([x[0] for x in sphere_coords]))
        my_convert_to_sofa(barycentric_output_path, config, row_angles, column_angles)
        print('Created barycentric baseline sofa files')

    config.path = config.barycentric_hrtf_dir
    file_ext = f'lsd_errors_barycentric_interpolated_data_{config.upscale_factor}.pickle'
    run_lsd_evaluation(config, barycentric_output_path, file_ext)

    file_ext = f'loc_errors_barycentric_interpolated_data_{config.upscale_factor}.pickle'
    run_localisation_evaluation(config, barycentric_output_path, file_ext)

    barycentric_data_folder = f'/barycentric_interpolated_data_{config.upscale_factor}'
    barycentric_output_path = config.barycentric_hrtf_dir + barycentric_data_folder
    cube, sphere = run_barycentric_interpolation(config, barycentric_output_path)

    if config.gen_sofa_flag:
        convert_to_sofa(barycentric_output_path, config, cube, sphere)
        print('Created barycentric baseline sofa files')

    config.path = config.barycentric_hrtf_dir
    #######################################################################################################################
    # # hrtf selection
    # hrtf_selection_output_path = 'C:/PycharmProjects/Upsample_GAN/ESRGAN_master/HRTF/baselines/hrtf_selection_data'
    # run_hrtf_selection(config, config.hrtf_selection_dir)
    # if config.gen_sofa_flag:
    #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #     ds = load_function(data_dir, feature_spec={'hrirs': {'samplerate': config.hrir_samplerate,
    #                                                          'side': 'left', 'domain': 'magnitude'}},
    #                        subject_ids='first')
    #     row_angles = ds.row_angles
    #     column_angles = ds.column_angles
    #     # my_convert_to_sofa(config.valid_gt_path, config, row_angles, column_angles)
    #     my_convert_to_sofa(config.hrtf_selection_dir, config, row_angles, column_angles)
    #
    # config.path = config.hrtf_selection_dir
    #
    # file_ext = f'lsd_errors_hrtf_selection_minimum_data.pickle'
    # run_lsd_evaluation(config, config.hrtf_selection_dir, file_ext, hrtf_selection='minimum')
    # file_ext = f'loc_errors_hrtf_selection_minimum_data.pickle'
    # run_localisation_evaluation(config, config.hrtf_selection_dir, file_ext, hrtf_selection='minimum')
    #
    # file_ext = f'lsd_errors_hrtf_selection_maximum_data.pickle'
    # run_lsd_evaluation(config, config.hrtf_selection_dir, file_ext, hrtf_selection='maximum')
    # file_ext = f'loc_errors_hrtf_selection_maximum_data.pickle'
    # run_localisation_evaluation(config, config.hrtf_selection_dir, file_ext, hrtf_selection='maximum')

if __name__ == '__main__':
    main()

