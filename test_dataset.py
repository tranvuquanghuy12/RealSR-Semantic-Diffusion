import yaml
from data import create_dataset
import options.options as option
import sys

try:
    opt = option.parse('options/isprs/train_Paper_Baseline.yml', is_train=True)
    val_opt = opt['datasets']['val']
    val_set = create_dataset(val_opt, 'SR')
    print(f"Len val_set: {len(val_set)}")
    if len(val_set) > 0:
        data = val_set[0]
        print(f"LQ shape: {data['LQ'].shape}")
        print(f"GT shape: {data['GT'].shape}")
except Exception as e:
    import traceback
    traceback.print_exc()
