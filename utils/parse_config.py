

def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')  # list. list中一个元素保存一行
    lines = [x for x in lines if x and not x.startswith('#')]  # 去掉注释行
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            module_defs.append({})  # 只要是'['开头，就新建一个往列表最后新插入一个字典，作为一个模块
            module_defs[-1]['type'] = line[1:-1].rstrip()  # 往列表最后的字典赋值
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")  # 不是'['开头，就往最后的字典内填充内容即可
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()  # 往列表最后一个元素（该元素是字典）赋值

    return module_defs


def parse_data_config(path):
    """Parses the data configuration file"""
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options
