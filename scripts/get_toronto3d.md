# Getting Toronto-3D

1. Download `Toronto_3D.zip` from the OneDrive link in https://github.com/WeikaiTan/Toronto-3D
   (CC BY-NC 4.0, about 1.1 GB). OneDrive needs a browser, so download it on your laptop.
2. Copy the four PLY files to the cluster:

   ```bash
   rsync -av L00*.ply <user>@<cluster-login-node>:/cluster/work/<user>/lidar-road-segmentation/data/toronto3d/
   ```

3. `python3 slurm/generate.py prepare --submit`
