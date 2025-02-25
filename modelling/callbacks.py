from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, RichModelSummary
from  pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.callbacks import LearningRateMonitor

def get_callbacks(callback_config):
   early_stop_callback = EarlyStopping(
      monitor=callback_config['EarlyStopping']['monitor'],
      min_delta=callback_config['EarlyStopping']['min_delta'],
      patience=callback_config['EarlyStopping']['patience'],
      verbose=True,
      mode=callback_config['EarlyStopping']['mode'],
   )

   theme = RichProgressBarTheme(metrics='green', time='yellow', progress_bar_finished='#8c53e0' ,
                                progress_bar='#c99e38')
   rich_progress_bar = RichProgressBar(theme=theme)

   rich_model_summary = RichModelSummary(max_depth=3)

   checkpoint_callback = ModelCheckpoint(
      monitor=callback_config['ModelCheckpoint']['monitor'],
      save_top_k=callback_config['ModelCheckpoint']['save_top_k'],
      mode=callback_config['ModelCheckpoint']['mode'],
      save_last=callback_config['ModelCheckpoint']['save_last'],
      verbose=True,
   )
   lr_monitor = LearningRateMonitor(logging_interval='step')

   return (early_stop_callback, checkpoint_callback, 
           rich_progress_bar, rich_model_summary, lr_monitor)