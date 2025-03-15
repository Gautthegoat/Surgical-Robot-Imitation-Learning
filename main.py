import source.engines as engines

import argparse
import os
import yaml
import datetime

def parse_arguments():
    parser = argparse.ArgumentParser(description="Automated Surgery Project")
    
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Choose the mode to run the script")

    # Training mode
    train_parser = subparsers.add_parser('train', help="Train the model")
    train_parser.add_argument('--engine', type=str, required=True, help="Name of the engine to use")
    train_parser.add_argument('--config', type=str, required=True, help="Name of the configuration file")

    # Resume training mode
    resume_parser = subparsers.add_parser('resume', help="Resume training from a checkpoint")
    resume_parser.add_argument('--archive_model', type=str, required=True, help="Model in Archive to resume training from")
    
    # Visualization mode
    visualize_parser = subparsers.add_parser('visualize', help="Visualize training results")
    visualize_parser.add_argument('--archive_model', type=str, required=True, help="Model in Archive to use for visualization")
    
    # Export mode
    export_parser = subparsers.add_parser('export', help="Export the model")
    export_parser.add_argument('--archive_model', type=str, required=True, help="Model in Archive to use for export")
    
    return parser.parse_args()

def main():
    args = parse_arguments()

    if args.mode == 'train':
        # Read the configuration file
        with open(f"source/configs/{args.config}.yaml", 'r') as file:
            config = yaml.safe_load(file)
        
        # Create the Archive folder
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_folder_name = f"{args.engine}_{config.get('model_name')}_{config.get('tag')}_{timestamp}"
        archive_folder_path = os.path.join('Archive/', archive_folder_name)
        os.makedirs(archive_folder_path, exist_ok=True)
        
        # Create the config file
        with open(os.path.join(archive_folder_path, 'config.yaml'), 'w') as file:
            yaml.dump(config, file)

        engine_name = args.engine

    else:
        # Check if the Archive folder exists
        archive_folder_path = os.path.join('Archive/', args.archive_model)
        if not os.path.exists(archive_folder_path):
            raise FileNotFoundError(f"Archive folder not found: {archive_folder_path}")
        engine_name = args.archive_model.split('/')[-1].split('_')[0]

    # Create the engine
    try:
        engine = getattr(engines, engine_name)(args, archive_folder_path)
    except Exception as e:
        raise ValueError(f"Engine {engine_name} not found. Error: {e}")

    # Run the engine based on the selected mode
    if args.mode == 'train':
        engine.train()
    elif args.mode == 'resume':
        engine.resume()
    elif args.mode == 'visualize':
        engine.visualize()
    elif args.mode == 'export':
        engine.export()

if __name__ == '__main__':
    main()

# Exemplary usage:
# python main.py train --config act_il --engine ClassicIL
# python main.py visualize --archive_model /home/gb/git/auto-suturing-ml/Archive/ClassicIL_ACTModel_Vanilla_20240907_183012
# python main.py export --archive_model /home/gb/git/auto-suturing-ml/Archive/ClassicIL_ACTModel_Vanilla_20240907_183012