from pathlib import Path

import click
from loguru import logger

from minimal_hand.utils import prepare_mano


MODEL_DIR = Path(__file__).parent.parent / "model"
HAND_MESH_MODEL_PATH = MODEL_DIR / "hand_mesh" / "hand_mesh_model.pkl"


@click.command()
@click.option("--mano",
              default=str(Path.home() / "dataset" / "smpl" / "mano_v1_2"), help="Path to the mano model")
def main(mano: str):
    logger.info("Verifying hand mesh model ...")
    prepare_mano(mano_dir=Path(mano), output_path=HAND_MESH_MODEL_PATH)


if __name__ == "__main__":
    main()
