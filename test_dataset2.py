import sys
import options.options as option
from data import create_dataset

try:
    opt = option.parse('options/isprs/train_Paper_Baseline.yml', is_train=True)
    opt = option.dict_to_nonedict(opt)
    val_opt = opt['datasets']['val']
    
    val_set = create_dataset(val_opt, 'SR')
    data = val_set[0]
    print(f"LQ shape: {data['LQ'].shape}")
    print(f"GT shape: {data['GT'].shape}")
except Exception as e:
    import traceback
    traceback.print_exc()
