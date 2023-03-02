import os
import sys
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from utils.preprocessor import TwitterDataModule
from models.finetune import Finetune


if __name__ == '__main__':

    seed_everything(seed=42, workers=True)

    parser = argparse.ArgumentParser(description='Trainer Parser')
    parser.add_argument('-m', '--model', choices=['IndoBERT', 'IndoBERTweet', 'IndoRoBERTa_OSCAR', 'IndoRoBERTa_Wiki'], required=True, help='Pretrinaed model choices to train')
    parser.add_argument('-lr', '--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size')

    args = parser.parse_args()
    config = vars(args)

    # Get arguments values
    model_name = config['model']
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']

    pretrained_model_name = {
        'IndoBERT': 'indolem/indobert-base-uncased',
        'IndoBERTweet': 'indolem/indobertweet-base-uncased',
        'IndoRoBERTa_OSCAR': 'cahya/roberta-base-indonesian-522M',
        'IndoRoBERTa_Wiki': 'flax-community/indonesian-roberta-base',
    }

    pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name[model_name])
    pretrained_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name[model_name], num_labels=2, output_attentions=False, output_hidden_states=False)

    data_module = TwitterDataModule(tokenizer=pretrained_tokenizer, batch_size=batch_size)
    model = Finetune(model=pretrained_model, learning_rate=learning_rate)

    # Initialize callbacks and progressbar
    tensor_board_logger = TensorBoardLogger('logs', name=f'{model_name}/{batch_size}_{learning_rate}')
    checkpoint_callback = ModelCheckpoint(dirpath=f'./checkpoints/{model_name}/{batch_size}_{learning_rate}', monitor='val_f1_score')
    early_stop_callback = EarlyStopping(monitor='val_f1_score', min_delta=0.00, check_on_train_epoch_end=1, patience=3)
    tqdm_progress_bar = TQDMProgressBar()

    # Initialize Trainer
    trainer = Trainer(
        accelerator='gpu',
        max_epochs=20,
        default_root_dir=f'./checkpoints/{model_name}/{batch_size}_{learning_rate}',
        callbacks=[checkpoint_callback, early_stop_callback, tqdm_progress_bar],
        logger=tensor_board_logger,
        deterministic=True  # To ensure reproducible results
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module, ckpt_path='best')