import numpy as np


def random(texture, num):
#    idx = np.random.choice(texture.shape[0], num, replace=False) # 乱数を抽出するときに重複を許さない場合(ただし、サンプル数が少ないとエラーになりやすい）
    idx = np.random.choice(texture.shape[0], num) # 乱数を抽出するときに重複を許す場合(ただし、サンプル数が少ない時でも安定）
    return texture[idx]


def stat(texture, num):
    pass


def hybrid(texture, num):
    pass


method = {'random': random, 'STAT': stat, 'HybridIA': hybrid}
