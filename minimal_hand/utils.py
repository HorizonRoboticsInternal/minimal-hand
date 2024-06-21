from pathlib import Path
import pickle

import numpy as np

from minimal_hand.kinematics import MANOHandJoints


def prepare_mano(mano_dir: Path,
                 output_path: Path,
                 mano_model_name: str = "MANO_LEFT.pkl"):
    with open(mano_dir / "models" / mano_model_name, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    output = {}
    output["verts"] = np.array(data["v_template"])
    output["faces"] = np.array(data["f"])
    output["mesh_basis"] = np.transpose(data["shapedirs"], (2, 0, 1))

    j_regressor = np.zeros([21, 778])
    j_regressor[:16] = data["J_regressor"].toarray()
    for k, v in MANOHandJoints.mesh_mapping.items():
        j_regressor[k, v] = 1
    output["j_regressor"] = j_regressor
    output["joints"] = np.matmul(output["j_regressor"], output["verts"])

    raw_weights = data["weights"]
    weights = [None] * 21
    weights[0] = raw_weights[:, 0]
    for j in "IMLRT":
        weights[MANOHandJoints.labels.index(j + "0")] = np.zeros(778)
        for k in [1, 2, 3]:
            src_idx = MANOHandJoints.labels.index(j + str(k - 1))
            tar_idx = MANOHandJoints.labels.index(j + str(k))
            weights[tar_idx] = raw_weights[:, src_idx]
    output["weights"] = np.expand_dims(np.stack(weights, -1), -1)
    with open(output_path, "wb") as f:
        pickle.dump(output, f)
