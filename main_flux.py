from optimization.image_editor import ImageEditor
from optimization.arguments import get_arguments
import os


if __name__ == "__main__":
    args = get_arguments()
    # Loop through images in input directory

    if args.cluster_path:
        # expect model_path to be path/to/cluster_model/modelNNNNNN.pt
        image_editor = ImageEditor(args)
        base_model_path = args.cluster_model_dir 
        for i in range(args.start_index, args.end_index + 1):
            path = None            
            for filename in os.listdir(os.path.join(base_model_path,f"cluster-{i}")):
                print(f"the filename is: {filename}")
                if filename.startswith(f"pytorch_"):
                    path = os.path.join(os.path.join(base_model_path,f"cluster-{i}"), filename)
            
            assert path is not None, f"Cluster {i} does not exist"
            #print(f"the path is: {path}")
            args.model_path = path
            image_editor.load_cluster_lora(path)
            image_editor.sample_image()
    #else:
        #image_editor = ImageEditor(args)
        #image_editor.sample_image()
