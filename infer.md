Basic terminology - 'take' == the original video, 'take_name' == the nickname you gave it.
Save the video you want to use as exo.mp4 in:
```
./
└── data
     └── videos
           └── take_name
                 └── exo.mp4
```

## Prior rendering (ViPE):

Run ViPE inference, results will be saved to `vipe_results/{take_name}`:
```bash
bash scripts/infer_vipe.sh
```

Run ViPE visualization tool to adjust the ego extrinsics by placing the arrows in the head position and in parallel to the gaze of the character:
```bash
vipe visualize vipe_results/{take_name}
```

Edit `generate_meta_json.py`: copy the extrinsics from the ViPE visualizer into ego_extrinsic and use the prompt template and your video to write a prompt (you can use mostly any AI chat for that). Then run the file:
```bash
python generate_meta_json.py
```
This will create a meta.json file in `data/vidoes`.
You can now terminate and close the ViPE visualizer.

Render the prior egocentric view by editing with the correct files and running:
```bash
sh script/render_vipe.sh
```
This will create a ego_Prior.mp4 video in `data/videos/{take_name}`

Convert the depth maps from the exo.zip file by running:
```bash
python scripts/convert_depth_zip_to_npy.py \
  --depth_path vipe_results/{take_name}/depth \
  --egox_depthmaps_path data/videos/depth_maps

# Rename the directory inside depth_maps from exo to take_name
mv data/videos/depth_maps/exo data/videos/depth_maps/{take_name}
```

## Egocentric video generation

Currently, `generate_meta_json.py` works only for a single video, so `infer.sh` works without changes for any video you will use (`meta.json` will contain only the last processed video and only it will be processed to egocentric view).


