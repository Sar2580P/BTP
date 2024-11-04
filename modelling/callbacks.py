from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, RichModelSummary
from  pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from utils import read_yaml

callback_configs = read_yaml('configs.yaml')['callbacks']

early_stop_callback = EarlyStopping(
   monitor=callback_configs['EarlyStopping']['monitor'],
   min_delta=callback_configs['EarlyStopping']['min_delta'],
   patience=callback_configs['EarlyStopping']['patience'],
   verbose=True,
   mode=callback_configs['EarlyStopping']['mode'],
)

theme = RichProgressBarTheme(metrics='green', time='yellow', progress_bar_finished='#8c53e0' ,progress_bar='#c99e38')
rich_progress_bar = RichProgressBar(theme=theme)

rich_model_summary = RichModelSummary(max_depth=3)

checkpoint_callback = ModelCheckpoint(
    monitor=callback_configs['ModelCheckpoint']['monitor'],
    save_top_k=callback_configs['ModelCheckpoint']['save_top_k'],
    mode=callback_configs['ModelCheckpoint']['mode'],
    save_last=callback_configs['ModelCheckpoint']['save_last'],
    verbose=True,
 )