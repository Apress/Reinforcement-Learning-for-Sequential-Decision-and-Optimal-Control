import numpy as np


def read_csv(path):
    import pandas as pd
    data = pd.read_csv(path)
    data_np = data.values
    return {'x': data_np[:, 0], 'y': data_np[:, 1]}


def read_np(path):
    data = np.load(path)
    return {'x': data[:, 0], 'y': data[:, 1]}


def save_csv(path, step, value):
    import pandas as pd
    df = pd.DataFrame({'Step': step, 'value': value})
    df.to_csv(path, index=False, sep=',')


def read_tensorboard(path, keys):
    """
    input the dir of the tensorboard log
    """
    from tensorboard.backend.event_processing import event_accumulator

    if isinstance(keys, str):
        keys = [keys]
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    print("All available keys in Tensorboard", ea.scalars.Keys())  # 我们知道tensorboard可以保存Image scalars等对象，我们主要关注scalars

    valid_key_list = [i for i in keys if i in ea.scalars.Keys()]
    assert len(valid_key_list) != 0, "invalid keys"

    output_dict = dict()
    for key in valid_key_list:
        event_list = ea.scalars.Items(key)
        x, y = [], []
        for e in event_list:
            x.append(e.step)
            y.append(e.value)

        data_dict = {'x': np.array(x), 'y': np.array(y)}
        output_dict[key] = data_dict
    return output_dict

if __name__ == '__main__':
    pass