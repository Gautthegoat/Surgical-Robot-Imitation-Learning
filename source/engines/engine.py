import os
import abc
import torch
import yaml
import logging
import matplotlib.pyplot as plt

class Engine(abc.ABC):
    def __init__(self, args, archive_folder_path):
        """
        Initialize the engine with common components like archive folder and logger.

        :param archive_folder_path: Path to the archive folder.
        :param logger: Logger instance for logging the processes
        """
        self.args = args
        self.archive_folder_path = archive_folder_path
        self.config_path = os.path.join(self.archive_folder_path, 'config.yaml')
        self.logger = self._setup_logging()
        self.config = self._load_config()
        self.start_epoch = 1

        self.checkpoints_dir = os.path.join(self.archive_folder_path, 'checkpoints')
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.logger.info("Engine initialized")

    @abc.abstractmethod
    def train(self):
        """
        Method to handle the training process.
        Must be implemented by any child engine.
        """
        raise NotImplementedError("The 'train' method must be implemented by the child engine.")

    def visualize(self):
        """
        Default method to visualize training results, can be overridden by child classes.
        This will visualize all metrics stored in the 'metrics.yaml' file, with x-axis as the steps (epochs).
        Child classes can override this method for custom visualization logic.
        """
        metrics_file = os.path.join(self.archive_folder_path, "metrics.yaml")
        
        if not os.path.exists(metrics_file):
            self.logger.error(f"Metrics file not found: {metrics_file}")
            raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
        
        # Load the metrics from the file
        with open(metrics_file, 'r') as file:
            metrics = yaml.safe_load(file)

        if not metrics:
            self.logger.error("No metrics found in the metrics file.")
            return

        # Create a new folder for plots
        plots_folder = os.path.join(self.archive_folder_path, "plots")
        os.makedirs(plots_folder, exist_ok=True)

        # Plot each metric dynamically
        for metric_name, values in metrics.items():
            steps = [entry['step'] for entry in values]
            metric_values = [entry['value'] for entry in values]

            plt.figure()
            plt.plot(steps, metric_values, label=metric_name)
            plt.xlabel('Steps')
            plt.ylabel(metric_name)
            plt.title(f'{metric_name.capitalize()} over Steps')
            plt.legend()

            # Save the plot
            plot_path = os.path.join(plots_folder, f'{metric_name}.png')
            plt.savefig(plot_path)
            self.logger.info(f"{metric_name.capitalize()} plot saved to {plot_path}")
    
    def resume(self):
        """
        Method to resume training from a checkpoint.
        """
        checkpoint_files = [f for f in os.listdir(self.checkpoints_dir) if f.startswith('checkpoint_epoch_')]
        # Sort the checkpoint files based on the epoch number
        checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        # Get the latest checkpoint file
        latest_checkpoint = checkpoint_files[-1]
        checkpoint_path = os.path.join(self.checkpoints_dir, latest_checkpoint)
        self.logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_metric = checkpoint['metric']
        self.start_epoch = checkpoint['epoch'] + 1
    
    @abc.abstractmethod
    def export(self):
        """
        Method to export the trained model.
        Must be implemented by any child engine.
        """
        raise NotImplementedError("The 'export' method must be implemented by the child engine.")


    def _load_config(self):
        """Load the config file for the engine."""
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.logger.info(f"Config loaded: {self.config_path}")
        return config

    def _save_checkpoint(self, model, optimizer, epoch, metric):
        """
        Save model and optimizer state at a specific checkpoint.

        :param model: Model instance to save.
        :param optimizer: Optimizer instance to save.
        :param epoch: Current epoch number.
        :param best_metric: Best metric (e.g., validation loss) recorded so far.
        """
        checkpoint_path = os.path.join(self.checkpoints_dir, f'checkpoint_epoch_{epoch}.pth')
        self.logger.info(f"Saving checkpoint at epoch {epoch}: {checkpoint_path}")
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'metric': metric
            
        }
        torch.save(checkpoint, checkpoint_path)

    # Only used for Initialization
    def _setup_logging(self):
        """
        Set up logging configuration to log to both a file in the archive folder and the console.
        
        :param archive_folder_path: Path to the archive folder where the log file will be created.
        :param log_filename: Name of the log file (default is 'train.log').
        :param log_level: Logging level (default is logging.INFO).
        :return: Configured logger instance.
        """
        # Log filename based on the mode
        log_filename = f"{self.args.mode}.log"
        
        # Create logger
        logger = logging.getLogger(self.archive_folder_path)  # Unique logger for each archive folder
        logger.setLevel(logging.DEBUG)
        
        # Ensure we don't add multiple handlers if setup_logging is called multiple times
        if logger.hasHandlers():
            logger.handlers.clear()

        # Create log directory if it doesn't exist
        log_file_path = os.path.join(self.archive_folder_path, "logs", log_filename)
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        # File handler to write logs to a file
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)

        # Console handler to also output logs to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # Log format
        log_format = logging.Formatter(
            "%(asctime)s - %(levelname)s - [%(module)s:%(funcName)s] - %(message)s"
        )
        file_handler.setFormatter(log_format)
        console_handler.setFormatter(log_format)

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logger.info(f"Logger initialized. Logging to {log_file_path}")

        return logger
    
    def _log_metrics(self, step, **kwargs):
        """
        Generic method to log metrics to a YAML file.
        """
        metrics_file = os.path.join(self.archive_folder_path, "metrics.yaml")

        # Load existing metrics if available
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = yaml.safe_load(f) or {}
        else:
            metrics = {}

        # Initialize empty lists for each metric if not already present
        for metric_name, metric_value in kwargs.items():
            if metric_name not in metrics:
                metrics[metric_name] = []

            # Append the step and corresponding metric value
            metrics[metric_name].append({'step': step, 'value': metric_value})

        # Save the metrics back to the file
        with open(metrics_file, 'w') as f:
            yaml.dump(metrics, f)      