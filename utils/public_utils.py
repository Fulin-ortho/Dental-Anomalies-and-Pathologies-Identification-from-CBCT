import os
import shutil
import uuid


def rename_file(old_path: str, new_path: str = '', exe: str = '.jpg'):
    """

    :param old_path: 原始文件夹路径
    :param new_path: 重命名后需要保存的文件路径
    :param exe: 需要更改的后缀
    :return: 无返回值
    """
    if not old_path:
        raise Exception("Invalid path!", old_path)
    if not os.path.isdir(old_path):
        raise Exception("Invalid path!", old_path)
    if new_path and not os.path.isdir(new_path):
        raise Exception("Invalid path!", new_path)

    path = old_path
    file_list = os.listdir(path)
    for index, file in enumerate(file_list):
        old_dir = os.path.join(path, file)

        _, extension = os.path.splitext(file)
        new_dir = os.path.join(
            new_path if new_path else path, str(index + 1) + exe)
        os.rename(old_dir, new_dir)
    print('end')


def print_network(net):
    """
    打印网络结构以及网络参数
    :param net: 需要打印的网络
    :return: 无返回值
    """
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def del_file(filepath):
    """
    删除某一目录下的所有文件或文件夹
    :param filepath: 路径
    :return:
    """
    try:
        del_list = os.listdir(filepath)
        for f in del_list:
            file_path = os.path.join(filepath, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    except:
        return -1
    return True


def uuid_name(name: str, type: int):
    """
    根据type生成不同类型的UUID
    :params name: 生成UUID的名
    :params type: 需要生成UUID的类型
    :return: 生成的UUI
    """
    if type == 1:
        return str(uuid.uuid1())
    elif type == 3:
        return str(uuid.uuid3(uuid.NAMESPACE_DNS, name))
    elif type == 4:
        return str(uuid.uuid4())
    elif type == 5:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, name))
