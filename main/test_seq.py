# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Modified from Keypoint Transformer, HFL-Net, DenseMutualAttention, and AlignSDF
# ------------------------------------------------------------------------------
#
# Frame-by-frame evaluation of a single sequence.
#
# Same eval path as main/test.py, but restricted to one sequence (e.g. "SB11" for
# HO3D, or "20200709-subject-01" for DexYCB) and printing/logging one line of
# metrics per frame instead of only the dataset-wide average.

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys

file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_dir + "/..")
import configargparse
import numpy as np
import tqdm
from torch.utils.data import DataLoader, Subset

from common.base import Tester
from common.metrics import eval_batched_obj_direct, eval_hand_joint
from common.utils.inverse_kinematics import ik_solver_mano
from common.utils.misc import *
from data.dataset_util import get_radius, prepare_model_template


def parse_args():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--gpu_ids", type=str, dest="gpu_ids", default="0")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        dest="ckpt_path",
        required=True,
        help="Full path to the checkpoint file",
    )
    parser.add_argument(
        "--seq_name",
        type=str,
        required=True,
        help="Sequence name to evaluate frame by frame (must be present in the "
        "evaluation split, e.g. 'evaluation.txt' for HO3D)",
    )
    parser.add_argument(
        "--temporal_window",
        type=int,
        default=5,
        help="Override cfg.temporal_window for evaluation. Use 0 to disable temporal input.",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default=None,
        help="Where to write per-frame results. Defaults to "
        "<ckpt_dir>/<seq_name>_results.txt",
    )
    args = parser.parse_args()

    if args.temporal_window is not None:
        cfg.temporal_window = args.temporal_window

    cfg.calc_mutliscale_dim(cfg.use_big_decoder, cfg.resnet_type)

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if "-" in args.gpu_ids:
        gpus = args.gpu_ids.split("-")
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ",".join(map(lambda x: str(x), list(range(*gpus))))

    return args


def main():
    args = parse_args()
    cfg.set_args(args.gpu_ids, args.ckpt_path)
    cfg.create_log_dir()

    tester = Tester(None)
    tester._make_batch_generator()

    full_set_list = tester.testset.set_list
    indices = [
        i for i, name in enumerate(full_set_list) if name.split("/")[0] == args.seq_name
    ]
    if len(indices) == 0:
        raise ValueError(
            "No frames found for sequence '{}' in the evaluation split. "
            "Available sequences: {}".format(
                args.seq_name,
                sorted(set(name.split("/")[0] for name in full_set_list)),
            )
        )
    indices.sort(key=lambda i: int(full_set_list[i].split("/")[1]))
    frame_names = [full_set_list[i] for i in indices]

    tester.batch_generator = DataLoader(
        dataset=Subset(tester.testset, indices),
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_thread,
        pin_memory=True,
    )

    templates, obj_names = prepare_model_template(cfg.simple_object_models_dir)
    radius = torch.from_numpy(np.array(get_radius(templates))).float()

    tester.ckpt_path = args.ckpt_path
    tester._make_model()

    ckpt_dir = os.path.dirname(args.ckpt_path)
    out_path = args.out_file or os.path.join(
        ckpt_dir, "{}_results.txt".format(args.seq_name)
    )
    log_file = open(out_path, "w+")

    if cfg.dataset == "dexycb":
        totals = {
            "ADDS_error": 0.0,
            "mano_mje": 0.0,
            "mano_pamje": 0.0,
            "OCE_error": 0.0,
            "MCE_error": 0.0,
        }
    else:
        totals = {"ADDS_error": 0.0, "MME_error": 0.0}
    total_samples = 0

    with torch.no_grad():
        for itr, (inputs_data, targets, meta_info) in enumerate(
            tqdm.tqdm(tester.batch_generator)
        ):
            frame_name = frame_names[itr]

            model_out = tester.model(inputs_data, targets, meta_info, "eval")
            out = {k[:-4]: model_out[k] for k in model_out.keys() if "_out" in k}

            ADDS_error, MCE_error, OCE_error, MME_error, sample_nums = (
                eval_batched_obj_direct(
                    out, targets, meta_info, templates, radius, obj_names
                )
            )

            if sample_nums == 0:
                # object class excluded from ADD-S/MME evaluation (e.g. "019_pitcher_base")
                line = "{}: skipped (excluded object class)".format(frame_name)
            else:
                total_samples += sample_nums
                parts = [
                    "{}:".format(frame_name),
                    "ADDS={:.3f}cm".format(ADDS_error * 100),
                ]
                totals["ADDS_error"] += ADDS_error * sample_nums * 100

                if cfg.dataset == "ho3d":
                    parts.append("MME={:.3f}cm".format(MME_error * 100))
                    totals["MME_error"] += MME_error * sample_nums * 100
                else:
                    parts.append("MCE={:.3f}cm".format(MCE_error * 100))
                    parts.append("OCE={:.3f}cm".format(OCE_error * 100))
                    totals["MCE_error"] += MCE_error * sample_nums * 100
                    totals["OCE_error"] += OCE_error * sample_nums * 100

                    if cfg.use_inverse_kinematics:
                        hand_joints = torch.cat(
                            [
                                torch.zeros_like(out["hand_joints"][:, :1]),
                                out["hand_joints"],
                            ],
                            dim=1,
                        )
                        hand_pose_results = ik_solver_mano(
                            out["mano_shape"], hand_joints
                        )
                        mano_mje, mano_pamje = eval_hand_joint(
                            hand_pose_results["joints"],
                            targets["joint_cam_no_trans"] / 1000,
                        )
                    else:
                        mano_mje, mano_pamje = eval_hand_joint(
                            out["mano_joints"], out["mano_joints_gt"]
                        )
                    parts.append("MJE={:.3f}cm".format(mano_mje * 100))
                    parts.append("PAMJE={:.3f}cm".format(mano_pamje * 100))
                    totals["mano_mje"] += mano_mje * sample_nums * 100
                    totals["mano_pamje"] += mano_pamje * sample_nums * 100

                line = " ".join(parts)

            print(line)
            print(line, file=log_file)

    print("\n---- {} average over {} samples ----".format(args.seq_name, total_samples))
    print(
        "\n---- {} average over {} samples ----".format(args.seq_name, total_samples),
        file=log_file,
    )
    for k in totals:
        avg = totals[k] / total_samples if total_samples else float("nan")
        print("{}: {}".format(k, avg))
        print("{}: {}".format(k, avg), file=log_file)

    log_file.close()
    print("\nWrote per-frame results to {}".format(out_path))


if __name__ == "__main__":
    main()
