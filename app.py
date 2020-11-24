# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

#Dashに必要なモジュールのインポート
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
#Dashのコールバック関数に必要なライブラリ
from dash.dependencies import Input, Output
import plotly.express as px
import numpy as np
import pandas as pd

from sym import equivalent
import conv
import extract
#グラフ描画のためのPlotlyライブラリ
import plotly.graph_objects as go
#訓練済みDNN-2Dによる応力-ひずみ曲線の推定 
import dnn2d_ss # <--------- 2020/11/18追加
#訓練済みDNN-3DによるYLD2000-2d降伏関数のパラメータ及び降伏曲面の推定
#import dnn3d_yld # <--------- 2020/11/19追加

from dash.exceptions import PreventUpdate #エラーの場合グラフをアップデートしない

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

colors = { 'background': '#282c2f', 'text': '#7FDBFF'}

#####################ここから山中研究室で開発したプログラム####################################

# アルミニウム合金の集合組織の優先方位を定義
preferred_orientation = {
    'S': np.concatenate((equivalent([59, 37, 63], False),
                         equivalent([53, 75, 34], False),
                         #equivalent([27, 58, 18], False),
                         #equivalent([32, 58, 18], False),
                         equivalent([48, 75, 34], False),
                         #equivalent([64, 37, 63], False)
                         )),
    'Goss':  np.concatenate((equivalent([90, 90, 45], False),
                             equivalent([ 0, 45,  0], False),
                             equivalent([ 0, 45, 90], False))),
    'Brass': np.concatenate((equivalent([35, 45, 90], False),
                             equivalent([55, 90, 45], False),
                             equivalent([35, 45,  0], False))),
    'Copper': np.concatenate((equivalent([90, 35, 45], False),
                              equivalent([39, 66, 27], False)))
}


# 3次元ガウス分布を生成
def gauss():
    d = np.random.multivariate_normal(
        [0, 0, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 1000)
    spl = np.array([-1.2815514e+00,
                    -8.41621007e-01,
                    -5.24401007e-01,
                    -2.53347007e-01,
                    0,
                    2.53346992e-01,
                    5.24400992e-01,
                    8.41620992e-01,
                    1.2815514e+00])
    for j in [0, 1, 2]:
        r = d[:, j]
        N = np.empty(10)
        for i in range(10):
            if i == 0:
                N[i] = len(np.where(r <= spl[0])[0])
            elif i == 9:
                N[i] = len(np.where(r > spl[8])[0])
            else:
                a = (np.where(r > spl[i - 1])[0])
                N[i] = len(np.where(r[a] <= spl[i])[0])
        n = np.sum((len(r) / 10 - N) ** 2 / (len(r) / 10))
        if n > 14.7:
            gauss()
    return d

# 疑似集合組織のデータを生成
class orientation(object):

    # 結晶の対称性を考慮して結晶方位を生成
    @staticmethod
    def unit(ss, c):
        s0 = 2. / 3.
        vol = 1000
        R = np.zeros([3, 3])
        if c == 0:
            R = np.array([[-1.414, 0, 1.414],
                          [0, 2, 0],
                          [-1.414, 0, -1.414]]) / 2
        else:
            R = np.array([[1.414, 0, 1.414],
                          [0, 2, 0],
                          [-1.414, 0, 1.414]]) / 2
        texture = np.zeros([int(vol * s0), 3])
        # note: np.random.rand() generates 0 to 1 random number
        texture[:, 0] = np.random.rand(int(vol * s0)) * 90 * 1.414
        gauss3d = gauss()
        texture[:, 1:] = gauss3d[:int(vol * s0), :2] * np.sqrt(ss)  # 67% : line
        texture = np.dot(texture, R) + np.array([90., 0., 0.])
        texture[texture[:, 1] < 0., 1] *= -1.
        texture[texture[:, 0] < 0., 0] += 90.
        texture[texture[:, 0] > 90., 0] -= 90.
        texture[texture[:, 2] < 0., 2] += 90.
        texture[texture[:, 2] > 90., 2] -= 90.
        texture0 = gauss3d[int(vol * s0):] * np.sqrt(ss)  # 33% : dot
        texture0[texture0[:, 1] > 0, 1] *= -1.
        texture0[(texture0[:, 0] < 0.) & (texture0[:, 2] > 0.)
                 ] += np.array([90., 0., 0.])
        texture0[(texture0[:, 0] < 0.) & (texture0[:, 2] < 0.)
                 ] += np.array([90., 0., 90.])
        texture0[texture0[:, 2] < 0] += np.array([0, 0, 90])
        texture0 += np.array([0, 90, 0])
        return np.concatenate((texture, texture0))


    # 3次元ガウス分布を用いて優先方位を生成
    @staticmethod
    def generate_pseudoTex(ss, vol, ori):
        """Method giving a three-dimensional Gaussian distribution to preferred orientation
        Arguments:
            ss {[type]} -- Dispersion angle
            vol {int} -- Volume fraction
            ori {array_like} -- Preferred orientation considering symmetry
        Returns:
            [array_like] -- preferred orientation with a three-dimensional Gaussian distribution
        """
        tmp = np.empty((0, 3))
        for i in range(ori.shape[0]):
            buf = gauss() * np.sqrt(ss) + ori[i]
            tmp = np.concatenate((tmp, buf))
        return extract.method['random'](tmp, int(vol + 0.5))


    # Bungeのオイラー角に変換
    @staticmethod
    def random(vol):
        return np.reshape(np.random.rand(int(vol + 0.5) * 3),(int(vol + 0.5), 3)) * [360, 180, 360]


    # 優先方位のCubeを定義
    @classmethod
    def cube(cls, ss, vol):
        ori = np.empty((0, 3))
        for i in range(4):
            for j in range(4):
                for k in range(2):
                    buf = cls.unit(ss, k)
                    if k == 0:
                        ori = np.concatenate((ori,buf + np.array([i * 90, 0, j * 90])))
                    else:
                        buf[:, 1] *= -1
                        buf += np.array([i * 90, 180, j * 90])
                        ori = np.concatenate((ori, buf))
        return extract.method['random'](ori, int(vol + 0.5))

    # 優先方位のGossを定義
    @classmethod
    def goss(cls, ss, vol):
        G = preferred_orientation['Goss']
        return cls.generate_pseudoTex(ss, int(vol + 0.5), G)

    # 優先方位のBrassを定義
    @classmethod
    def brass(cls, ss, vol):
        B = preferred_orientation['Brass']
        return cls.generate_pseudoTex(ss, int(vol + 0.5), B)

    # 優先方位のCopperを定義
    @classmethod
    def copper(cls, ss, vol):
        Cu = preferred_orientation['Copper']
        return cls.generate_pseudoTex(ss, int(vol + 0.5), Cu)

    # 優先方位のSを定義
    @classmethod
    def S(cls, ss, vol):
        S = preferred_orientation['S']
        return cls.generate_pseudoTex(ss, int(vol + 0.5), S)


# 疑似集合組織の極点図の画像を生成
class Texture_old(object):
    def __init__(self, volume=1000):
        self.tex_data = np.empty((0, 3))
        self.vol = volume
        self.vol_max = 20000

    def __clamp(self, val, max_val=255, min_val=0):
        val[val > max_val] = max_val
        val[val < min_val] = min_val
        return val

    def add(self, texture):
        self.tex_data = np.concatenate((self.tex_data, texture))

    def addRandom(self, percentage,addvolume):
        self.add(orientation.random(addvolume * percentage / 100.))

    def addCube(self, ss, percentage,addvolume):
        self.add(orientation.cube(ss, addvolume * percentage / 100.))

    def addS(self, ss, percentage,addvolume):
        self.add(orientation.S(ss, addvolume * percentage / 100.))

    def addGoss(self, ss, percentage,addvolume):
        self.add(orientation.goss(ss, addvolume * percentage / 100.))

    def addBrass(self, ss, percentage,addvolume):
        self.add(orientation.brass(ss, addvolume * percentage / 100.))

    def addCopper(self, ss, percentage,addvolume):
        self.add(orientation.copper(ss, addvolume * percentage / 100.))

    def fromTxt(self, file_name, num_header):
        txtData = np.loadtxt(file_name, skiprows=num_header, usecols=(0, 1, 2))
        self.add(txtData)

    # 極点図の画像を生成　デフォルトでは(111)極点図
    def pole_figure(self, direct=np.array([1, 1, 1]),
                    denominator=10, img_size=128, method='random'):
        """Positive pole figure

        Keyword Arguments:
            direct {array_like} -- Direction of projection plane (default: {np.array([1, 1, 1])})
            denominator {int} -- N (default: {10})
            img_size {int} -- image size (default: {128})
            method {str} -- Extraction method (default: {'random'})

        Returns:
            numpy array (img_size, img_size) -- pole figure ss image data
        """
        stereo = conv.stereo(self.sample(method), direct)
        stereo = np.array([stereo / 2. * img_size / 2. +
                           img_size / 2.], dtype="int32")
        img = np.zeros([img_size, img_size])
        for i, j in stereo[0]:
            if denominator == 0:
                img[i, j] = 1
            else:
                img[i, j] += 1
        img = img / denominator * 255.0 if denominator != 0 else img * 255
        img = self.__clamp(img)
        return img

    def sample(self, method='random'):
        """Extract crystal orientation

        Keyword Arguments:
            method {str} -- Extract method (default: {'random'})

        Returns:
            [numpy array] -- [description]
        """
        tmp = extract.method[method](self.tex_data, self.vol)
        tmp[tmp < 0.] += 360.
        tmp[tmp > 360.] -= 360.
        return tmp

    # ステレオ投影の極点を算出
    def projection_point(self, direct=np.array([1, 1, 1]), method='random'):
        stereo = conv.stereo(self.sample(method), direct)
        return stereo

    # 生成した疑似集合組織の数値データをファイル保存
    def saveTxt(self, file_name, method='random'):
        data_to_save = []
        append = data_to_save.append
        for euler in self.sample(method):
            tmp = [euler[0], euler[1], euler[2], 1.0]
            append(tmp)
        header = 'B 2000'
        np.savetxt(file_name, np.array(data_to_save), header=header, comments='', delimiter='\t',
                   fmt=['%3.5f', '%3.5f', '%3.5f', '%3.5f'])

    # 生成した疑似集合組織の極点図(デフォルト(111))を画像ファイルとして出力・保存
    def savePoleFigure(self, file_name, direct=np.array([1, 1, 1]), invert=True,
                       denominator=10, img_size=128, method='random'):
        """Positive pole figure

        Keyword Arguments:
            direct {array_like} -- Direction of projection plane (default: {np.array([1, 1, 1])})
            denominator {int} -- N (default: {10})
            img_size {int} -- image size (default: {128})
            method {str} -- Extraction method (default: {'random'})

        Returns:
            numpy array (img_size, img_size) -- pole figure ss image data
        """
        stereo = conv.stereo(self.sample(method), direct)
        print(stereo[:,0],stereo[:,1])
        stereo = np.array([stereo / 2. * img_size / 2. +
                           img_size / 2.], dtype="int32")
        img = np.zeros([img_size, img_size])
        for i, j in stereo[0]:
            if denominator == 0:
                img[i, j] = 1
            else:
                img[i, j] += 1
        img = img / denominator * 255.0 if denominator != 0 else img * 255
        img = self.__clamp(img)
        pil = Image.fromarray(np.uint8(img))
        if invert:
            pil = ImageOps.invert(pil)
        pil.save(file_name)

    # Plotlyを使って極点図を描画(デフォルトは(111)極点図）
    def Plotly_PoleFigure(self, direct=np.array([1, 1, 1]), method='random', component='Unknown'):
        """Positive pole figure

        Keyword Arguments:
            direct {array_like} -- Direction of projection plane (default: {np.array([1, 1, 1])})
            method {str} -- Extraction method (default: {'random'})

        Returns:
            numpy array (img_size, img_size) -- pole figure ss image data
        """
        stereo = conv.stereo(self.sample(method), direct)
        layout = go.Layout(width=500,height=500,xaxis=dict(range=[-2,2]),yaxis=dict(range=[-2,2]))  
        fig = go.Figure(data=go.Scatter(x=stereo[:,0], y=stereo[:,1], mode='markers'), layout=layout)
        return fig
###################################ここまで################################################


##########################ここからはwebアプリケーション部分######################################

def create_input_card(title, subtitle_id, body):
    card = dbc.Card(dbc.CardBody([
        html.H5(title, className="card-title"),
        html.H6(dbc.FormText(id=subtitle_id), className="card-subtitle"),
        html.P(body,
            className="card-text",
            style={"margin-top":"5px"})
        ]), style={"width":"100%", "margin":"2px"})
    return card

card_pole_type = create_input_card(
    title="極点図の種類", 
    subtitle_id="radioitems-input-status",
    body=dbc.RadioItems(
        options=[
            {'label': '{100}', 'value': '1,0,0'},
            {'label': '{110}', 'value': '1,1,0'},
            {'label': '{111}', 'value': '1,1,1'}
        ],
        value='1,1,1', #初期値を設定
        id="radioitems-input",
    )
)
card_num_crystal_orientation = create_input_card(
    title='結晶方位の数(0～5000)',
    subtitle_id="number-of-orientation-status",
    body=dbc.Input(
        id='number-of-orientation',
        type='number',
        placeholder="input number of crystal orientation ...",
        value='1000' #初期値を設定
    )
)
card_vol_cube_texture = create_input_card(
    title='Cube方位の体積分率(0～100)',
    subtitle_id="volume-fraction-cube-status",
    body=dbc.Input(
        id='volume-fraction-cube',
        type='number',
        placeholder="input volume fraction of Cube texture ...",
        value='0' #初期値を設定
    )
)
card_vol_brass_texture = create_input_card(
    title='Brass方位の体積分率(0～100)',
    subtitle_id="volume-fraction-brass-status",
    body=dbc.Input(
        id='volume-fraction-brass',
        type='number',
        placeholder="input volume fraction of Brass texture ...",
        value='0' #初期値を設定
    )
)
card_vol_goss_texture = create_input_card(
    title='Goss方位の体積分率(0～100)',
    subtitle_id="volume-fraction-goss-status",
    body=dbc.Input(
        id='volume-fraction-goss',
        type='number',
        placeholder="input volume fraction of Goss texture ...",
        value='0' #初期値を設定
    )
)
card_vol_s_texture = create_input_card(
    title='S方位の体積分率(0～100)',
    subtitle_id="volume-fraction-s-status",
    body=dbc.Input(
        id='volume-fraction-s',
        type='number',
        placeholder="input volume fraction of S texture ...",
        value='0' #初期値を設定
    )
)
card_vol_copper_texture = create_input_card(
    title='Copper方位の体積分率(0～100)',
    subtitle_id="volume-fraction-copper-status",
    body=dbc.Input(
        id='volume-fraction-copper',
        type='number',
        placeholder="input volume fraction of Copper texture ...",
        value='0' #初期値を設定
    )
)

app.title = "Numerical Material Test on Web"
app.layout = dbc.Container(
    [
        dbc.Row(html.H2(className="title", children='Numerical Material Test on Web')),
        dbc.Row(html.P(children='Yamanaka research group @ TUAT')),

        dbc.Row(     # dbc: dash bootstrap components
        [
            dbc.Col(
            [
                dbc.Row(card_pole_type),
                dbc.Row(card_num_crystal_orientation),
                dbc.Row(card_vol_cube_texture),
                dbc.Row(card_vol_brass_texture),
                dbc.Row(card_vol_goss_texture),
                dbc.Row(card_vol_s_texture),
                dbc.Row(card_vol_copper_texture),
            ], width=3),
            
            dbc.Col(
                dcc.Graph(id='pole-figure'),  # dcc: dash core components
#                dcc.Graph(id='stress-strain'),
            ),

            dbc.Col( 
                dcc.Graph(id='stress-strain'),  # <------ 2020/11/18 追加: 応力-ひずみ曲線を描画
#                dcc.Graph(id='yield-surface'),  # <------ 2020/11/19 追加: 降伏曲面を描画
            )
        ]
    ),
])

# コールバック関数群とそれに対応する関数の定義
@app.callback(
    dash.dependencies.Output('radioitems-input-status', 'children'),
    [dash.dependencies.Input('radioitems-input', 'value')])
def update_output(value):
    return 'Type of pole figure = {} '.format(value)

@app.callback(
    dash.dependencies.Output('number-of-orientation-status', 'children'),
    [dash.dependencies.Input('number-of-orientation', 'value')])
def update_output(value):
    return 'Number of crystal orientation = {} '.format(value)

@app.callback(
    dash.dependencies.Output('volume-fraction-cube-status', 'children'),
    [dash.dependencies.Input('volume-fraction-cube', 'value')])
def update_output(value):
    return 'Volume fraction of Cube texture = {} [%]'.format(value)

@app.callback(
    dash.dependencies.Output('volume-fraction-brass-status', 'children'),
    [dash.dependencies.Input('volume-fraction-brass', 'value')])
def update_output(value):
    return 'Volume fraction of Brass texture = {} [%]'.format(value)

@app.callback(
    dash.dependencies.Output('volume-fraction-goss-status', 'children'),
    [dash.dependencies.Input('volume-fraction-goss', 'value')])
def update_output(value):
    return 'Volume fraction of Goss texture = {} [%]'.format(value)

@app.callback(
    dash.dependencies.Output('volume-fraction-s-status', 'children'),
    [dash.dependencies.Input('volume-fraction-s', 'value')])
def update_output(value):
    return 'Volume fraction of S texture = {} [%]'.format(value)

@app.callback(
    dash.dependencies.Output('volume-fraction-copper-status', 'children'),
    [dash.dependencies.Input('volume-fraction-copper', 'value')])
def update_output(value):
    return 'Volume fraction of Copper texture = {} [%]'.format(value)

@app.callback(
    Output(component_id='pole-figure',component_property='figure'),
    [
     Input(component_id='radioitems-input', component_property='value'),
     Input(component_id='number-of-orientation', component_property='value'),
     Input(component_id='volume-fraction-cube', component_property='value'),
     Input(component_id='volume-fraction-brass', component_property='value'),
     Input(component_id='volume-fraction-goss', component_property='value'),
     Input(component_id='volume-fraction-s', component_property='value'),
     Input(component_id='volume-fraction-copper', component_property='value')])

def update_figure(radioitems_input_str,num_ori_str,vf_Cu_str,vf_Br_str,vf_Go_str,vf_S_str,vf_Co_str):
    if vf_Cu_str is None:
        raise PreventUpdate
    if vf_Br_str is None:
        raise PreventUpdate
    if vf_Go_str is None:
        raise PreventUpdate
    if vf_S_str is None:
        raise PreventUpdate
    if vf_Co_str is None:
        raise PreventUpdate

    num_ori = int(num_ori_str)
    vf_Cu = int(vf_Cu_str)
    vf_Br = int(vf_Br_str)
    vf_Go = int(vf_Go_str)
    vf_S  = int(vf_S_str)
    vf_Co = int(vf_Co_str)

    tex  = Texture_old(volume=num_ori)
    addvolume = num_ori
    pref_tex = ['Cube','Brass','Goss','S','Copper','Random']
    pref_var = [    10,     10,    10,   10,    10,      10] #cube方位の分散5～15deg2
    pref_vol = [ vf_Cu,  vf_Br, vf_Go, vf_S, vf_Co,       0] 
    pref_vol[5] = 100-pref_vol[0]-pref_vol[1]-pref_vol[2]-pref_vol[3]-pref_vol[4]
    tex.addCube(  pref_var[0], pref_vol[0],addvolume)  # input Variance & Volume fraction
    tex.addBrass( pref_var[1], pref_vol[1],addvolume)
    tex.addGoss(  pref_var[2], pref_vol[2],addvolume)
    tex.addS(     pref_var[3], pref_vol[3],addvolume)
    tex.addCopper(pref_var[4], pref_vol[4],addvolume)
    tex.addRandom(             pref_vol[5],addvolume)

    radioitems_str = radioitems_input_str.split(',')
    radioitems_int = [int(n) for n in radioitems_str]
    #print(radioitems_int)

    fig = tex.Plotly_PoleFigure(direct=np.array(radioitems_int),component='Added')
    #fig = tex.Plotly_PoleFigure(component='Added')
    
    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'],
        transition_duration=500)

    return fig


# 2020/11/18 追加: 入力された数値に対して, 応力ひずみ曲線を計算し描画
@app.callback(
    Output(component_id='stress-strain',component_property='figure'),
    [
     Input(component_id='volume-fraction-cube', component_property='value'),
     Input(component_id='volume-fraction-brass', component_property='value'),
     Input(component_id='volume-fraction-goss', component_property='value'),
     Input(component_id='volume-fraction-s', component_property='value'),
     Input(component_id='volume-fraction-copper', component_property='value')])
def update_ss_curve(vf_Cu_str,vf_Br_str,vf_Go_str,vf_S_str,vf_Co_str):
    if vf_Cu_str is None:
        raise PreventUpdate
    if vf_Br_str is None:
        raise PreventUpdate
    if vf_Go_str is None:
        raise PreventUpdate
    if vf_S_str is None:
        raise PreventUpdate
    if vf_Co_str is None:
        raise PreventUpdate

    vf_Cu = int(vf_Cu_str)
    vf_Br = int(vf_Br_str)
    vf_Go = int(vf_Go_str)
    vf_S  = int(vf_S_str)
    vf_Co = int(vf_Co_str)
    texture = '0_0{0:02d}13_0{1:02d}08_0{2:02d}12_0{3:02d}13_0{4:02d}05'.format(vf_Cu,vf_Br,vf_Go,vf_S,vf_Co)

    roots = ['1_0', '4_1', '2_1', '4_3', '1_1', '3_4', '1_2', '1_4', '0_1']

    strain, stress, std = dnn2d_ss.estimate_sscurve(roots, texture, 5)

    layout = go.Layout(width=500,height=500)
    fig = go.Figure(layout=layout)
    fig.update_xaxes(title_text='True stress for RD [MPa]')
    fig.update_yaxes(title_text='True stress for TD [MPa]')
    fig.add_trace(go.Scatter(x=strain[5,:,0], y=stress[5,:,0],name="RD"))    
    fig.add_trace(go.Scatter(x=strain[5,:,1], y=stress[5,:,1],name="TD"))
    
    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'],
       transition_duration=500)
    
    return fig

'''
# 2020/11/19 追加: 訓練済みDNN-3dを用いてYld200-2dの降伏曲面を計算し描画
@app.callback(
    Output(component_id='yield-surface',component_property='figure'),
    [
     Input(component_id='volume-fraction-cube', component_property='value'),
     Input(component_id='volume-fraction-brass', component_property='value'),
     Input(component_id='volume-fraction-goss', component_property='value'),
     Input(component_id='volume-fraction-s', component_property='value'),
     Input(component_id='volume-fraction-copper', component_property='value')])
def update_yield_surface(vf_Cu_str,vf_Br_str,vf_Go_str,vf_S_str,vf_Co_str):
    if vf_Cu_str is None:
        raise PreventUpdate
    if vf_Br_str is None:
        raise PreventUpdate
    if vf_Go_str is None:
        raise PreventUpdate
    if vf_S_str is None:
        raise PreventUpdate
    if vf_Co_str is None:
        raise PreventUpdate

    vf_Cu = int(vf_Cu_str)
    vf_Br = int(vf_Br_str)
    vf_Go = int(vf_Go_str)
    vf_S  = int(vf_S_str)
    vf_Co = int(vf_Co_str)
    texture = '0_0{0:02d}13_0{1:02d}08_0{2:02d}12_0{3:02d}13_0{4:02d}05'.format(vf_Cu,vf_Br,vf_Go,vf_S,vf_Co)

    stand = [0.001, 0.01, 0.02, 0.03, 0.04] # 基準塑性ひずみ

    ALFA, ref = dnn3d_yld.estimateYldParam(stand, texture, 50)  # パラメータa_1~8とM
    print('Estimated parameters alpha_1~8 & M: ', ALFA[4,:])

    layout = go.Layout(width=500,height=500,yaxis=dict(scaleanchor='x'))
    fig = go.Figure(layout=layout)
    fig.update_xaxes(title_text='True stress for RD [MPa]')
    fig.update_yaxes(title_text='True stress for TD [MPa]')
    cnt = 0
    for lps in stand:
        strain = '{}'.format(lps)
        yld = dnn3d_yld.yldSurface(ALFA[cnt], ref[cnt])
        fig.add_trace(go.Scatter(x=yld[:, 0], y=yld[:, 1], name=strain))
        cnt += 1

    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'],
        transition_duration=500)
    return fig
'''


if __name__ == '__main__':
    app.run_server(debug=True)
