import json
import os
import shutil
import subprocess
from pathlib import Path

def process_scenes():
    # Read the train_index.json
    train_index_path = "/data/sudheerbabu/oct/models/3/datasets/dl3dv-sample/test_index1.json"
    with open(train_index_path, 'r') as f:
        train_data = json.load(f)
    
    # Iterate through each scene
    for i, scene in enumerate(train_data):
        print(f"\n{'='*60}")
        print(f"Processing scene {i}: {scene}")
        print(f"{'='*60}\n")
        
        # 1) Create temp_in directory
        temp_dir = Path("/data/sudheerbabu/oct/models/4/feats/temp_feats")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 2) Copy the scene_i directory
        scene_name = scene  # Assuming scene is the directory name
        source_scene = Path(f"/data/sudheerbabu/oct/models/3/datasets/dl3dv-sample/{scene_name}")
        dest_scene = temp_dir / scene_name
        
        if source_scene.exists():
            shutil.copytree(source_scene, dest_scene, dirs_exist_ok=True)
            print(f"Copied {source_scene} to {dest_scene}")
        else:
            print(f"Warning: Source scene {source_scene} does not exist. Skipping...")
            shutil.rmtree(temp_dir)
            continue
        
        # 3) Create train_index.json and test_index.json in temp_in
        train_index_content = [scene]
        test_index_content = [scene]
        
        with open(temp_dir / "train_index.json", 'w') as f:
            json.dump(train_index_content, f, indent=2)
        
        with open(temp_dir / "test_index.json", 'w') as f:
            json.dump(test_index_content, f, indent=2)
        
        print(f"Created train_index.json and test_index.json in {temp_dir}")
        #3.1) Create temp_out directory for saving features
        temp_out_dir = Path("/data/sudheerbabu/oct/models/4/feats/aggregator_feats") / scene_name
        temp_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Created output directory at {temp_out_dir}")
        # 4) Run the python command
        command = (
            "CUDA_VISIBLE_DEVICES=0 python src/main.py "
            "+experiment=dl3dv "
            "trainer.num_nodes=1 "
            "dataset.dl3dv.view_sampler.num_context_views=3 "
            "dataset.dl3dv.view_sampler.num_target_views=3 "
            "dataset.dl3dv.view_sampler.use_fixed_indices=true "
            "dataset.dl3dv.fixed_context_indices=[0,8,20] "
            "dataset.dl3dv.fixed_target_indices=[0,8,20] "
            "dataset.dl3dv.roots=[/data/sudheerbabu/oct/models/4/feats/temp_feats] "
            "dataset.dl3dv.augment=false "
            "dataset.dl3dv.intr_augment=false "
            "dataset.dl3dv.input_image_shape=[448,448] "
            "model.encoder.distill=true "
            "dataset.dl3dv.view_sampler.max_img_per_gpu=3 "
            "wandb.mode=disabled "
            "model.encoder.pretrained_features=true "
            "model.encoder.freeze_aggregator=true "
            "model.encoder.freeze_backbone=false "
            "optimizer.warm_up_steps=10 "
            "model.encoder.freeze_module=all "
            "model.encoder.save_feats=true "
            "model.encoder.comp_aggregator=true "
        )
        
        print(f"Running command: {command}\n")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Error running command for scene {i}: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
        # 4) Move the output file to the temp_out directory
        if temp_out_dir.exists():
            shutil.move("/data/sudheerbabu/oct/models/4/feats/temp_feats/aggregated_list.pt", temp_out_dir / "aggregated_list.pt")
            # shutil.move("/data/sudheerbabu/oct/models/4/feats/temp_feats/input_image.pt", temp_out_dir / "input_image.pt")
        # 5) Delete the temp_in directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"\nDeleted {temp_dir}")
        
        print(f"\nCompleted processing scene {i}")

if __name__ == "__main__":
    process_scenes()