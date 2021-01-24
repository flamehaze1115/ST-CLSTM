import os
import torch
import torch.utils.data.dataloader
import numpy as np
import cv2


def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def delete_if_exist(path):
    if os.path.exists(path):
        os.remove(path)


def cubes_2_maps(cubes):
    b, c, d, h, w = cubes.shape
    cubes = cubes.permute(0, 2, 1, 3, 4)

    return cubes.contiguous().view(b * d, c, h, w), b, d


def maps_2_cubes(maps, b, d):
    x_b, x_c, x_h, x_w = maps.shape
    maps = maps.contiguous().view(b, d, x_c, x_h, x_w)

    return maps.permute(0, 2, 1, 3, 4)


def colorize_depth_np(input, max_depth, color_mode=cv2.COLORMAP_RAINBOW):
    input_tensor = input
    input_tensor[input_tensor > max_depth] = max_depth
    normalized = input_tensor / max_depth * 255.0
    normalized = normalized.astype(np.uint8)
    if len(input_tensor.shape) == 3:
        normalized_color = np.zeros((input_tensor.shape[0],
                                     input_tensor.shape[1],
                                     input_tensor.shape[2],
                                     3))
        for i in range(input_tensor.shape[0]):
            normalized_color[i] = cv2.applyColorMap(normalized[i], color_mode)
        return normalized_color
    if len(input_tensor.shape) == 2:
        normalized = cv2.applyColorMap(normalized, color_mode)
        return normalized


def inference(model, test_loader, device, metrics_s=None, output_dir=""):
    model.eval()
    if metrics_s is not None: metrics_s.reset()

    with torch.no_grad():
        count = 0
        for image, depth, depth_scaled, test_indices, save_depthfiles in test_loader:
            image = image.to(device)

            output = model(image)
            # output = maps_2_cubes(output, batch_size, depth_size)

            for id, index in enumerate(test_indices):
                save_depthfile = save_depthfiles[id]
                save_dir = os.path.join(output_dir, save_depthfile.split("/")[-3], "pred_depth")
                make_if_not_exist(save_dir)

                pred_depth = output[id, :, index].cpu().numpy()
                np.save(os.path.join(save_dir, os.path.basename(save_depthfile).replace("png", "npy")),
                        pred_depth)
                colored_depth = colorize_depth_np(pred_depth, max_depth=5.)
                cv2.imwrite(os.path.join(save_dir, os.path.basename(save_depthfile)),
                            colored_depth)

        #         if metrics_s is not None:
        #             metrics_s(torch.stack(output_new, 0).cpu(), torch.stack(depth_new, 0))
        #
        # result_s = metrics_s.loss_get()
        # print(result_s)
