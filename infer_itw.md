# From DGX, pwd=/work/exo_to_ego
srun -p mig,work -G 1 --mem 100000 --container-image ./exo_to_ego.sqsh --container-mounts ./:/workspace --pty bash

# From inside the container, pwd=/workspace
conda activate exo_to_ego
bash EgoX-main/scripts/infer_itw.sh
