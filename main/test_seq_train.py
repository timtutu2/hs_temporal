# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Modified from Keypoint Transformer, HFL-Net, DenseMutualAttention, and AlignSDF
# ------------------------------------------------------------------------------
#
# Frame-by-frame evaluation of a single sequence from the HO3D *train* split.
#
# main/test_seq.py only works on sequences listed in evaluation.txt, whose meta
# .pkl files carry no hand-joint ground truth (HO3D's official test split is
# submission-only). Train-split meta .pkl files DO carry full 21-joint
# "handJoints3D" (plus MANO pose/shape), so this script additionally reports
# per-frame hand MJE/PA-MJE, not just the object ADD-S/MME metrics.
#
# It deliberately does not reuse data/ho3d.py's Dataset(mode="train") path,
# since that path is built for training (random augmentation + SDF point
# sampling) rather than deterministic eval. Instead it re-implements the same
# non-augmented crop/preprocess steps data/ho3d.py uses for its eval split,
# pointed at the train/ folder.

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys

file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_dir + "/..")
import configargparse
import numpy as np
import torchvision.transforms as transforms
import tqdm
from PIL import Image
from torch.utils.data import DataLoader

from common.base import Tester
from common.metrics import eval_batched_obj_direct, eval_hand_joint
from common.utils.misc import *
from data import dataset_util, ho3d_util
from data.dataset_util import convert_pose_to_opencv, get_center_cam
from data.dataset_util import get_radius, prepare_model_template

jointsMapManoToSimple = [
    0,
    13,
    14,
    15,
    16,
    1,
    2,
    3,
    17,
    4,
    5,
    6,
    18,
    10,
    11,
    12,
    19,
    7,
    8,
    9,
    20,
]
jointsMapSimpleToMano = np.argsort(jointsMapManoToSimple)

COORD_CHANGE_MAT = np.array(
    [[1.0, 0.0, 0.0], [0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32
)


class HO3DTrainSeqDataset(torch.utils.data.Dataset):
    """Deterministic (no augmentation) loader for one HO3D train-split
    sequence, preprocessed the same way as data/ho3d.py's eval split so it can
    be fed straight into the model in "eval" mode."""

    def __init__(self, seq_name):
        self.root = cfg.ho3d_data_dir
        self.seq_name = seq_name
        self.inp_res = cfg.input_img_shape[0]
        self.obj_depth_mean_value = cfg.obj_depth_mean_value
        self.transform = transforms.ToTensor()

        self.obj_mesh = ho3d_util.load_objects_HO3D(cfg.object_models_dir)
        self.obj_bbox3d = dataset_util.get_bbox21_3d_from_dict(self.obj_mesh)

        rgb_dir = os.path.join(self.root, "train", seq_name, "rgb")
        if not os.path.isdir(rgb_dir):
            raise ValueError("No such train sequence directory: " + rgb_dir)
        self.frame_ids = sorted(
            os.path.splitext(fname)[0]
            for fname in os.listdir(rgb_dir)
            if fname.endswith(".png")
        )

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        frame_id = self.frame_ids[idx]
        img_path = os.path.join(self.root, "train", self.seq_name, "rgb", frame_id + ".png")
        img = Image.open(img_path).convert("RGB")
        annotations = np.load(
            os.path.join(self.root, "train", self.seq_name, "meta", frame_id + ".pkl"),
            allow_pickle=True,
        )

        K = np.array(annotations["camMat"], dtype=np.float32)
        obj_bbox3d = self.obj_bbox3d[annotations["objName"]]
        obj_pose = ho3d_util.pose_from_RT(
            annotations["objRot"].reshape((3,)), annotations["objTrans"]
        )
        p2d = ho3d_util.projectPoints(obj_bbox3d, K, rt=obj_pose)

        # train meta .pkl carries the full 21-joint hand GT (unlike the
        # eval-split .pkl, which only has the root joint under the same key).
        joints_3d_gt = np.array(annotations["handJoints3D"], dtype=np.float32)
        root_joint = joints_3d_gt[0].copy().dot(COORD_CHANGE_MAT.T)

        # eval-split .pkl files ship a precomputed "handBoundingBox"; train
        # .pkl files don't, so derive an equivalent tight box from the GT
        # joints instead (same source data/ho3d.py's train augmentation uses
        # for its own hand crop box).
        joints_uv = ho3d_util.projectPoints(joints_3d_gt, K)
        bbox_hand = dataset_util.get_bbox_joints(joints_uv, bbox_factor=1.0)

        img, K, bbox_hand, bbox_obj, _ = self._data_crop(img, K, bbox_hand, p2d)

        obj_center_cam = get_center_cam(bbox_obj, self.obj_depth_mean_value, K).astype(
            np.float32
        )

        img = self.transform(np.asarray(img).astype(np.float32)) / 255.0
        obj_rot, obj_trans = convert_pose_to_opencv(
            annotations["objRot"].squeeze(), annotations["objTrans"]
        )
        obj_trans = obj_trans.astype(np.float32) - obj_center_cam

        obj_mask = (
            annotations["objName"] == "021_bleach_cleanser"
            or annotations["objName"] == "006_mustard_bottle"
            or annotations["objName"] == "010_potted_meat_can"
        )

        inputs = {"img": img}
        targets = {
            "obj_rot": obj_rot,
            "rel_obj_trans": obj_trans.astype(np.float32),
            # raw (untransformed) GT, in the same camera-frame convention the
            # model's predicted joints end up in after the coord_change_mat
            # round-trip test.py applies -- see eval loop below.
            "joints_3d_gt_full": joints_3d_gt,
        }
        meta_info = {
            "cam_intr": K,
            "mano_root": root_joint,
            "hand_type": "right",
            "obj_cls": annotations["objName"],
            "obj_mask": obj_mask,
            "obj_center_cam": obj_center_cam,
            "bbox_hand": bbox_hand.astype(np.float32),
            "bbox_obj": bbox_obj.astype(np.float32),
        }
        return inputs, targets, meta_info

    def _data_crop(self, img, K, bbox_hand, p2d):
        crop_hand = dataset_util.get_bbox_joints(bbox_hand.reshape(2, 2), bbox_factor=1.5)
        crop_obj = dataset_util.get_bbox_joints(p2d, bbox_factor=1.5)
        bbox_hand = dataset_util.get_bbox_joints(bbox_hand.reshape(2, 2), bbox_factor=1.2)
        bbox_obj = dataset_util.get_bbox_joints(p2d, bbox_factor=1.0)
        center, scale = dataset_util.fuse_bbox(crop_hand, crop_obj, img.size)
        affinetrans, _ = dataset_util.get_affine_transform(
            center, scale, [self.inp_res, self.inp_res]
        )
        bbox_hand = dataset_util.transform_coords(
            bbox_hand.reshape(2, 2), affinetrans
        ).flatten()
        bbox_obj = dataset_util.transform_coords(
            bbox_obj.reshape(2, 2), affinetrans
        ).flatten()
        img = dataset_util.transform_img(img, affinetrans, [self.inp_res, self.inp_res])
        img = img.crop((0, 0, self.inp_res, self.inp_res))
        K = affinetrans.dot(K)
        return img, K, bbox_hand, bbox_obj, None


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
        help="Train-split sequence name to evaluate frame by frame, e.g. 'BB10'",
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
        "<ckpt_dir>/<seq_name>_train_results.txt",
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
    if cfg.dataset != "ho3d":
        raise ValueError(
            "test_seq_train.py only supports cfg.dataset == 'ho3d' "
            "(cfg.setting is currently '{}')".format(cfg.setting)
        )

    cfg.set_args(args.gpu_ids, args.ckpt_path)
    cfg.create_log_dir()

    seq_dataset = HO3DTrainSeqDataset(args.seq_name)
    frame_names = [
        "{}/{}".format(args.seq_name, frame_id) for frame_id in seq_dataset.frame_ids
    ]

    tester = Tester(None)
    tester.batch_generator = DataLoader(
        dataset=seq_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_thread,
        pin_memory=True,
    )
    tester.jointsMapSimpleToMano = jointsMapSimpleToMano

    templates, obj_names = prepare_model_template(cfg.simple_object_models_dir)
    radius = torch.from_numpy(np.array(get_radius(templates))).float()

    tester.ckpt_path = args.ckpt_path
    tester._make_model()

    ckpt_dir = os.path.dirname(args.ckpt_path)
    out_path = args.out_file or os.path.join(
        ckpt_dir, "{}_train_results.txt".format(args.seq_name)
    )
    log_file = open(out_path, "w+")

    totals = {"ADDS_error": 0.0, "MME_error": 0.0, "MJE_error": 0.0, "PAMJE_error": 0.0}
    total_samples = 0

    with torch.no_grad():
        for itr, (inputs_data, targets, meta_info) in enumerate(
            tqdm.tqdm(tester.batch_generator)
        ):
            frame_name = frame_names[itr]

            model_out = tester.model(inputs_data, targets, meta_info, "eval")
            out = {k[:-4]: model_out[k] for k in model_out.keys() if "_out" in k}

            ADDS_error, _, _, MME_error, sample_nums = eval_batched_obj_direct(
                out, targets, meta_info, templates, radius, obj_names
            )

            # Predicted hand joints, brought into the same raw camera-frame
            # convention as the untransformed GT (see class docstring/comment
            # above for the coord_change_mat round-trip this mirrors from
            # main/test.py's pred_mano.json export).
            mano_joints = out["mano_joints"].detach().cpu().numpy()
            mano_joints = mano_joints + meta_info["mano_root"].cpu().numpy()[:, None, :]
            mano_joints = np.matmul(mano_joints, COORD_CHANGE_MAT)
            mano_joints = mano_joints[:, jointsMapSimpleToMano]

            mano_mje, mano_pamje = eval_hand_joint(
                torch.from_numpy(mano_joints), targets["joints_3d_gt_full"]
            )
            totals["MJE_error"] += mano_mje * 100
            totals["PAMJE_error"] += mano_pamje * 100

            if sample_nums == 0:
                line = "{}: MJE={:.3f}cm PAMJE={:.3f}cm (object skipped, excluded class)".format(
                    frame_name, mano_mje * 100, mano_pamje * 100
                )
            else:
                total_samples += sample_nums
                totals["ADDS_error"] += ADDS_error * sample_nums * 100
                totals["MME_error"] += MME_error * sample_nums * 100
                line = "{}: ADDS={:.3f}cm MME={:.3f}cm MJE={:.3f}cm PAMJE={:.3f}cm".format(
                    frame_name,
                    ADDS_error * 100,
                    MME_error * 100,
                    mano_mje * 100,
                    mano_pamje * 100,
                )

            print(line)
            print(line, file=log_file)

    num_frames = len(frame_names)
    summary = "\n---- {} average over {} frames ({} used for ADDS/MME) ----".format(
        args.seq_name, num_frames, total_samples
    )
    print(summary)
    print(summary, file=log_file)
    for k in ("ADDS_error", "MME_error"):
        avg = totals[k] / total_samples if total_samples else float("nan")
        print("{}: {}".format(k, avg))
        print("{}: {}".format(k, avg), file=log_file)
    for k in ("MJE_error", "PAMJE_error"):
        avg = totals[k] / num_frames if num_frames else float("nan")
        print("{}: {}".format(k, avg))
        print("{}: {}".format(k, avg), file=log_file)

    log_file.close()
    print("\nWrote per-frame results to {}".format(out_path))


if __name__ == "__main__":
    main()
