_base_ = ['./petr_wifi.py']

evaluation = dict(interval=5, metric='mpjpe', save_best='mpjpe', rule='less')
checkpoint_config = dict(interval=5, max_keep_ckpts=10)
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
auto_resume = True
