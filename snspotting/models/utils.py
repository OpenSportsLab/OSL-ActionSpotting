import os
import logging
def create_folders(split, work_dir, overwrite):
    # Create folder name and zip file name
    output_folder=f"results_spotting_{'_'.join(split)}"
    output_results=os.path.join(work_dir, f"{output_folder}.zip")
    
    # Prevent overwriting existing results
    if os.path.exists(output_results) and not overwrite:
        logging.warning("Results already exists in zip format. Use [overwrite=True] to overwrite the previous results.The inference will not run over the previous results.")
        stop_predict=True
        # return output_results
    return output_folder, output_results, True if stop_predict else False