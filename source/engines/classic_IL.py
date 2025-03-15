from source.engines.engine import Engine
import source.datasets as datasets
import source.models as models

import os
import torch
import yaml
from torch.utils.data import DataLoader

class ClassicIL(Engine):
    def __init__(self, archive_folder_path, logger):
        super(ClassicIL, self).__init__(archive_folder_path, logger)
        self.device = torch.device(self.config['device'])

    def _load_data(self):
        """
        Load the training and validation data.
        """
        self.logger.info("Loading datasets...")

        try:
            dataset = getattr(datasets, self.config['dataset'])
        except AttributeError:
            raise NotImplementedError(f"Dataset {self.config['dataset']} not implemented")

        train_dataset = dataset(self.config, self.archive_folder_path, self.logger, mode='train')
        val_dataset = dataset(self.config, self.archive_folder_path, self.logger, mode='val', norm_stats=train_dataset.norm_stats)

        self.train_loader = DataLoader(train_dataset, 
                                       batch_size=self.config['data']['batch_size'], 
                                       shuffle=self.config['data']['shuffle'], 
                                       num_workers=self.config['data']['num_workers'], 
                                       pin_memory=self.config['data']['pin_memory'])

        self.val_loader = DataLoader(val_dataset, 
                                     batch_size=self.config['data']['batch_size'],
                                     shuffle=False, 
                                     num_workers=self.config['data']['num_workers'], 
                                     pin_memory=self.config['data']['pin_memory'])

        self.logger.info(f"Training data: {len(train_dataset)} samples, Validation data: {len(val_dataset)} samples")

    def _setup_training(self):
        """
        Setup the optimizer and loss function for training.
        """
        self.logger.info("Setting up training parameters...")

        # Setup the optimizer
        if self.config['training']['optimizer'] == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['training']['learning_rate'], weight_decay=self.config['training']['weight_decay'])
        else:
            raise ValueError(f"Error when setting up optimizer")

        self.best_metric = float('inf')
        self.nb_epochs = self.config['training']['epochs']

    def _setup_model(self):
        """
        Configure the model, optimizer, and move model to the appropriate device.
        """
        self.logger.info("Building the model...")

        model_name = self.config['model_name']
        try:
            self.model = getattr(models, model_name)(self.config).to(self.device)
        except:
            raise NotImplementedError(f"Model {model_name} not implemented")
        self.logger.info(f"Model: {model_name}, nb of parameters: {sum(p.numel() for p in self.model.parameters())}")

    def _evaluate(self, epoch):
        """
        Evaluate the model on the validation set.
        """
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, target in self.val_loader:
                inputs = [input_tensor.to(self.device) for input_tensor in inputs]
                target = target.to(self.device)
                predictions = self.model(inputs)
                loss = self.model.criterion_evaluation(predictions, target)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(self.val_loader)
        self.logger.info(f"Validation loss after epoch {epoch}: {avg_val_loss}")

        self._log_metrics(epoch, val_loss=avg_val_loss)

        # Save the per joints errors average of the batch       
        errors = torch.abs(predictions[0] - target)
        avg_errors = torch.mean(torch.mean(errors, dim=0), dim=0)
        self._log_metrics(
            epoch,
            val_error_j1_L=avg_errors[0].item(),
            val_error_j2_L=avg_errors[1].item(),
            val_error_j3_L=avg_errors[2].item(),
            val_error_j4_L=avg_errors[3].item(),
            val_error_j5_L=avg_errors[4].item(),
            val_error_j6_L=avg_errors[5].item(),
            val_error_j1_R=avg_errors[6].item(),
            val_error_j2_R=avg_errors[7].item(),
            val_error_j3_R=avg_errors[8].item(),
            val_error_j4_R=avg_errors[9].item(),
            val_error_j5_R=avg_errors[10].item(),
            val_error_j6_R=avg_errors[11].item()
        )

        # Save the best model based on validation loss
        if avg_val_loss < self.best_metric:
            self.logger.info(f"New best model found! Validation loss improved from {self.best_metric} to {avg_val_loss}.")
            self.best_metric = avg_val_loss
            self._save_checkpoint(self.model, self.optimizer, epoch, self.best_metric)

    def train(self):
        """
        Training process.
        """
        self.logger.info("""
 _____          _       _                   
|_   _| __ __ _(_)_ __ (_)_ __   __ _       
  | || '__/ _` | | '_ \| | '_ \ / _` |      
  | || | | (_| | | | | | | | | | (_| |_ _ _ 
  |_||_|  \__,_|_|_| |_|_|_| |_|\__, (_|_|_)
                                |___/       
        """)

        # Load data and setup model autonomously
        self._load_data()
        self._setup_model()
        self._setup_training()

        for epoch in range(self.start_epoch, self.nb_epochs + 1):
            self.logger.info(f"Epoch {epoch}/{self.nb_epochs}")
            self.model.train()
            total_loss = 0

            for idx, (inputs, target) in enumerate(self.train_loader):
                inputs = [input_tensor.to(self.device) for input_tensor in inputs]
                target = target.to(self.device)

                # Forward pass
                predictions = self.model(inputs)
                loss = self.model.criterion_training(predictions, target)

                # Backward pass and optimizer step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(self.train_loader)
            self.logger.info(f"Average training loss for epoch {epoch}: {avg_train_loss}")

            # Log metrics for visualization
            self._log_metrics(epoch, train_loss=avg_train_loss)

            # Evaluate the model on validation data after each epoch
            self._evaluate(epoch)

            # Save checkpoints
            if epoch % self.config['checkpoint']['save_freq'] == 0:
                self._save_checkpoint(self.model, self.optimizer, epoch, avg_train_loss)

    def export(self):
        # TODO
        pass
