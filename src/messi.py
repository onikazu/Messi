# 行動空間は離散的にしてあるA3C

import player11
import threading

import tensorflow as tf
import time, random, threading

import keras as kr
from keras.models import *
from keras.layers import *
from keras.utils import plot_model
from keras import backend as K
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # TensorFlow高速化用のワーニングを表示させない

# -- constants of Game
ENV = 'CartPole-v0'
# env = gym.make(ENV)
NUM_STATES = 4    # CartPoleは4状態
NUM_ACTIONS = 5       # CartPoleは、右に左に押す2アクション
ACTIONS = ("(turn 0)", "(turn 60)", "(turn -60)", "(dash 100)", "(kick 100 0)")
NONE_STATE = np.zeros(NUM_STATES)

# -- constants of LocalBrain
MIN_BATCH = 5
LOSS_V = .5  # v loss coefficient
LOSS_ENTROPY = .01  # entropy coefficient
LEARNING_RATE = 5e-3
RMSPropDecaly = 0.99

# -- params of Advantage-ベルマン方程式
GAMMA = 0.99
N_STEP_RETURN = 5
GAMMA_N = GAMMA ** N_STEP_RETURN

N_WORKERS = 8   # スレッドの数
Tmax = 10   # 各スレッドの更新ステップ間隔

# ε-greedyのパラメータ
EPS_START = 0.5
EPS_END = 0.0
EPS_STEPS = 200*N_WORKERS

# 全環境下で共有するステップ数
frames = 0

# 1episode が終わったかどうか確認するフラグ
isLearned = False

# --グローバルなTensorFlowのDeep Neural Networkのクラスです　-------
class ParameterServer:
    def __init__(self):
        with tf.variable_scope("parameter_server"):      # スレッド名で重み変数に名前を与え、識別します（Name Space）
            self.model = self._build_model()            # ニューラルネットワークの形を決定

        # serverのパラメータを宣言
        self.weights_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="parameter_server")
        self.optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, RMSPropDecaly)    # loss関数を最小化していくoptimizerの定義です

    # 関数名がアンダースコア2つから始まるものは「外部から参照されない関数」、「1つは基本的に参照しない関数」という意味
    def _build_model(self):     # Kerasでネットワークの形を定義します

        if os.path.isfile("./models/param_server_model.hdf5"):
            model = load_model("./models/param_server_model.hdf5")
            return model
        l_input = Input(batch_shape=(None, NUM_STATES))
        l_dense = Dense(16, activation='relu')(l_input)
        out_actions = Dense(NUM_ACTIONS, activation='softmax')(l_dense)
        out_value = Dense(1, activation='linear')(l_dense)
        model = Model(inputs=[l_input], outputs=[out_actions, out_value])
        plot_model(model, to_file='A3C.png', show_shapes=True)  # Qネットワークの可視化
        return model


# --各スレッドで走るTensorFlowのDeep Neural Networkのクラスです　-------
class LocalBrain:
    def __init__(self, name, parameter_server):   # globalなparameter_serverをメンバ変数として持つ
        with tf.name_scope(name):
            self.train_queue = [[], [], [], [], []]  # s, a, r, s', s' terminal mask
            K.set_session(SESS)
            self.model = self._build_model()  # ニューラルネットワークの形を決定
            self._build_graph(name, parameter_server)  # ネットワークの学習やメソッドを定義

    def _build_model(self):     # Kerasでネットワークの形を定義します
        l_input = Input(batch_shape=(None, NUM_STATES))
        l_dense = Dense(16, activation='relu')(l_input)
        out_actions = Dense(NUM_ACTIONS, activation='softmax')(l_dense)
        out_value = Dense(1, activation='linear')(l_dense)
        model = Model(inputs=[l_input], outputs=[out_actions, out_value])
        model._make_predict_function()  # have to initialize before threading
        return model

    def _build_graph(self, name, parameter_server):      # TensorFlowでネットワークの重みをどう学習させるのかを定義します
        self.s_t = tf.placeholder(tf.float32, shape=(None, NUM_STATES))  # placeholderは変数が格納される予定地となります
        self.a_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
        self.r_t = tf.placeholder(tf.float32, shape=(None, 1))  # not immediate, but discounted n step reward

        p, v = self.model(self.s_t)

        # loss関数を定義します
        log_prob = tf.log(tf.reduce_sum(p * self.a_t, axis=1, keep_dims=True) + 1e-10)
        advantage = self.r_t - v
        loss_policy = - log_prob * tf.stop_gradient(advantage)  # stop_gradientでadvantageは定数として扱います
        loss_value = LOSS_V * tf.square(advantage)  # minimize value error
        entropy = LOSS_ENTROPY * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1, keep_dims=True)  # maximize entropy (regularization)
        self.loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

        # 重みの変数を定義
        self.weights_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)  # パラメータを宣言
        # 勾配を取得する定義
        self.grads = tf.gradients(self.loss_total, self.weights_params)

        # ParameterServerの重み変数を更新する定義(zipで各変数ごとに計算)
        self.update_global_weight_params = \
            parameter_server.optimizer.apply_gradients(zip(self.grads, parameter_server.weights_params))

        # PrameterServerの重み変数の値を、localBrainにコピーする定義
        self.pull_global_weight_params = [l_p.assign(g_p)
                                          for l_p, g_p in zip(self.weights_params, parameter_server.weights_params)]

        # localBrainの重み変数の値を、PrameterServerにコピーする定義
        self.push_local_weight_params = [g_p.assign(l_p)
                                          for g_p, l_p in zip(parameter_server.weights_params, self.weights_params)]

    def pull_parameter_server(self):  # localスレッドがglobalの重みを取得する
        SESS.run(self.pull_global_weight_params)

    def push_parameter_server(self):  # localスレッドの重みをglobalにコピーする
        SESS.run(self.push_local_weight_params)

    def update_parameter_server(self):     # localbrainの勾配でParameterServerの重みを学習・更新します
        if len(self.train_queue[0]) < MIN_BATCH:    # データがたまっていない場合は更新しない
            return

        s, a, r, s_, s_mask = self.train_queue
        self.train_queue = [[], [], [], [], []]
        s = np.vstack(s)    # vstackはvertical-stackで縦方向に行列を連結、いまはただのベクトル転置操作
        a = np.vstack(a)
        r = np.vstack(r)
        s_ = np.vstack(s_)
        s_mask = np.vstack(s_mask)

        # Nステップあとの状態s_から、その先得られるであろう時間割引総報酬vを求めます
        _, v = self.model.predict(s_)

        # N-1ステップあとまでの時間割引総報酬rに、Nから先に得られるであろう総報酬vに割引N乗したものを足します
        r = r + GAMMA_N * v * s_mask  # set v to 0 where s_ is terminal state
        feed_dict = {self.s_t: s, self.a_t: a, self.r_t: r}     # 重みの更新に使用するデータ
        SESS.run(self.update_global_weight_params, feed_dict)   # ParameterServerの重みを更新

    def predict_p(self, s):    # 状態sから各actionの確率pベクトルを返します
        p, v = self.model.predict(s)
        return p

    def train_push(self, s, a, r, s_):
        self.train_queue[0].append(s)
        self.train_queue[1].append(a)
        self.train_queue[2].append(r)

        if s_ is None:
            self.train_queue[3].append(NONE_STATE)
            self.train_queue[4].append(0.)
        else:
            self.train_queue[3].append(s_)
            self.train_queue[4].append(1.)

# Agentクラス(ライブラリのようなもの)
class Messi():
    def __init__(self, name, parameter_server):
        super(Messi, self).__init__()
        # ワーカースレッド数
        self.n_workers = 1

        # 貪欲法のパラメータ
        self.eps_start = 0.5
        self.eps_end = 0.0
        self.eps_steps = 200 * self.n_workers

        self.brain = LocalBrain(name, parameter_server)  # 行動を決定するための脳（ニューラルネットワーク）
        self.memory = []  # s,a,r,s_の保存メモリ、　used for n_step return
        self.R = 0.  # 時間割引した、「いまからNステップ分あとまで」の総報酬R

    def act(self, s):
        """
        行動を決定する関数
        :param s:(list) situation
        :return:(int) action number which should be executed
        """
        global frames
        if frames >= self.eps_steps:   # ε-greedy法で行動を決定します 171115修正
            eps = self.eps_end
        else:
            eps = self.eps_start + frames * (self.eps_end - self.eps_start) / self.eps_steps  # linearly interpolate

        if random.random() < eps:
            return random.randint(0, NUM_ACTIONS - 1)   # ランダムに行動
        else:
            s = np.array([s])
            p = self.brain.predict_p(s)

            # a = np.argmax(p)  # これだと確率最大の行動を、毎回選択

            # a 0~4 が出力される
            a = np.random.choice(NUM_ACTIONS, p=p[0])
            # probability = p のこのコードだと、確率p[0]にしたがって、行動を選択
            # pにはいろいろな情報が入っていますが確率のベクトルは要素0番目
            return a

    def advantage_push_local_brain(self, s, a, r, s_):   # advantageを考慮したs,a,r,s_をbrainに与える
        def get_sample(memory, n):  # advantageを考慮し、メモリからnステップ後の状態とnステップ後までのRを取得する関数
            s, a, _, _ = memory[0]
            _, _, _, s_ = memory[n - 1]
            return s, a, self.R, s_

        # one-hotコーディングにしたa_catsをつくり、、s,a_cats,r,s_を自分のメモリに追加
        a_cats = np.zeros(NUM_ACTIONS)  # turn action into one-hot representation
        a_cats[a] = 1
        self.memory.append((s, a_cats, r, s_))

        # 前ステップの「時間割引Nステップ分の総報酬R」を使用して、現ステップのRを計算
        self.R = (self.R + r * GAMMA_N) / GAMMA     # r0はあとで引き算している、この式はヤロミルさんのサイトを参照

        # advantageを考慮しながら、LocalBrainに経験を入力する
        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_ = get_sample(self.memory, n)
                self.brain.train_push(s, a, r, s_)
                self.R = (self.R - self.memory[0][2]) / GAMMA
                self.memory.pop(0)

            self.R = 0  # 次の試行に向けて0にしておく

        if len(self.memory) >= N_STEP_RETURN:
            s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
            self.brain.train_push(s, a, r, s_)
            self.R = self.R - self.memory[0][2]     # # r0を引き算
            self.memory.pop(0)


# agent+env
class Environment(player11.Player11):
    total_reward_vec = np.zeros(10)  # 総報酬を10試行分格納して、平均総報酬をもとめる
    count_trial_each_thread = int(sys.argv[1])     # 各環境の試行数

    def __init__(self, name, thread_type, parameter_server):
        super(Environment, self).__init__()
        self.name = name
        self.thread_type = thread_type
        # self.env = gym.make(ENV)
        self.reward = 0
        self.agent = Messi(name, parameter_server)    # 環境内で行動するagentを生成

        for i in range(1):
            teamname = "Messi"
            if i < 11:
                teamname += "left"
            else:
                teamname += "right"
            self.initialize((i % 11 + 1), teamname, "localhost", 6000)
            self.start()

    def run(self):
        self.agent.brain.pull_parameter_server()  # ParameterSeverの重みを自身のLocalBrainにコピー
        global frames  # セッション全体での試行数、global変数を書き換える場合は、関数内でglobal宣言が必要です
        global isLearned
        global SESS
        global parameter_server

        # if (self.thread_type is 'test') and (self.count_trial_each_thread == 0):
        #     self.env.reset()
        #     self.env = gym.wrappers.Monitor(self.env, './movie/A3C')  # 動画保存する場合

        s = [self.m_dX, self.m_dY, self.m_dBallX, self.m_dBallY]
        a = random.randint(0, NUM_ACTIONS - 1)
        R = 0
        step = 0

        while True:
            message = self.receive()
            # print(message)
            # 初期メッセージの処理
            # print("p11:message:", message)
            if message.startswith("(init "):
                self.analyzeInitialMessage(message)
            # 視覚メッセージの処理
            elif message.startswith("(see "):
                self.analyzeVisualMessage(message)
            # 体調メッセージの処理
            elif message.startswith("(sense_body "):
                self.analyzePhysicalMessage(message)
                if self.m_iVisualTime < self.m_iTime:
                    self.predict(self.m_iVisualTime, self.m_iTime)

                if self.checkInitialMode():
                    if self.checkInitialMode():
                        self.setKickOffPosition()
                        command = \
                            "(move " + str(self.m_dKickOffX) + " " + str(self.m_dKickOffY) + ")"
                        self.m_strCommand = command

                if self.m_strPlayMode.startswith("play_on"):

                    s_ = [self.m_dX, self.m_dY, self.m_dBallX, self.m_dBallY]

                    self.calc_reward()
                    r = self.reward

                    # Advantageを考慮した報酬と経験を、localBrainにプッシュ
                    self.agent.advantage_push_local_brain(s, a, r, s_)

                    # 状態関数の更新
                    R += r
                    s = s_

                    if step % Tmax == 0:  # 終了時がTmaxごとに、parameterServerの重みを更新し、それをコピーする
                        if self.thread_type is 'learning':
                            self.agent.brain.update_parameter_server()
                            self.agent.brain.pull_parameter_server()

                    a = self.agent.act(s)
                    # コマンド実行部分
                    self.m_strCommand = ACTIONS[a]
                    self.send(self.m_strCommand)
                    frames += 1

                    # 1000step まで行っていたら1episode終了
                    if frames == 1000:
                        isLearned = True
                        break
                self.send(self.m_strCommand)

            # 聴覚メッセージの処理
            elif message.startswith("(hear "):
                self.analyzeAuralMessage(message)
            # サーバパラメータの処理
            elif message.startswith("(server_param"):
                self.analyzeServerParam(message)
            # プレーヤーパラメータの処理
            elif message.startswith("(player_param"):
                self.analyzePlayerParam(message)
            # プレーヤータイプの処理
            elif message.startswith("(player_type"):
                self.analyzePlayerType(message)
                # print("player_type_message", message)
            # エラーの処理
            else:
                print("p11 サーバーからエラーが伝えられた:", message)
                print("p11 エラー発生原因のコマンドは右記の通り :", self.m_strCommand)

        # 総試行数、スレッド名、今回の報酬を出力
        print("スレッド：" + self.name + "、試行数：" + str(self.count_trial_each_thread) + "、今回のステップ:" + str(
            step) + "、平均ステップ：" + str(self.total_reward_vec.mean()))

        # スレッドで平均報酬が一定を越えたら終了
        if self.total_reward_vec.mean() > 199:
            isLearned = True
            time.sleep(2.0)     # この間に他のlearningスレッドが止まります
            self.agent.brain.push_parameter_server()    # この成功したスレッドのパラメータをparameter-serverに渡します

        # 学習済みの重みを保存
        parameter_server.model.save('./models/param_server_model.hdf5')

        print("save is completed!!!")


    def calc_reward(self):
        self.reward = 0
        if self.m_strCommand.startswith("(kick") and self.getDistance(self.m_dX, self.m_dY, self.m_dBallX, self.m_dBallY) < 0.5:
            self.reward += 1

        if self.m_strSide.startswith("r"):
            if self.m_strPlayMode.startswith("goal_r"):
                self.reward += 1000

        if self.m_strSide.startswith("l"):
            if self.m_strPlayMode.startswith("goal_l"):
                self.reward += 1000


# --スレッドになるクラスです　-------
class Worker_thread:
    # スレッドは学習環境environmentを持ちます
    def __init__(self, thread_name, thread_type, parameter_server):
        self.environment = Environment(thread_name, thread_type, parameter_server)
        self.thread_type = thread_type

    def run(self):
        while True:
            if not(isLearned) and self.thread_type is 'learning':     # learning threadが走る
                self.environment.run()

            if not(isLearned) and self.thread_type is 'test':    # test threadを止めておく
                time.sleep(1.0)

            if isLearned and self.thread_type is 'learning':     # learning threadを止めておく
                time.sleep(3.0)

            if isLearned and self.thread_type is 'test':     # test threadが走る
                time.sleep(3.0)
                self.environment.run()


if __name__ == "__main__":
    # saver = tf.train.Saver()

    SESS = tf.Session()

    # M1.スレッドを作成します
    with tf.device("/cpu:0"):
        parameter_server = ParameterServer()  # 全スレッドで共有するパラメータを持つエンティティです
        threads = []  # 並列して走るスレッド
        # 学習するスレッドを用意
        for i in range(N_WORKERS):
            thread_name = "local_thread" + str(i + 1)
            threads.append(
                Worker_thread(thread_name=thread_name, thread_type="learning", parameter_server=parameter_server))
    # M2.TensorFlowでマルチスレッドを実行します
    COORD = tf.train.Coordinator()  # TensorFlowでマルチスレッドにするための準備です
    SESS.run(tf.global_variables_initializer())  # TensorFlowを使う場合、最初に変数初期化をして、実行します

    running_threads = []
    for worker in threads:
        job = lambda: worker.run()  # この辺は、マルチスレッドを走らせる作法だと思って良い
        t = threading.Thread(target=job)
        t.start()

    #
    # plays = []
    # for i in range(4):
    #     p = Messi()
    #     plays.append(p)
    #     teamname = str(p.__class__.__name__)
    #     if i < 11:
    #         teamname += "left"
    #     else:
    #         teamname += "right"
    #     plays[i].initialize((i % 2 + 1), teamname, "localhost", 6000)
    #     plays[i].start()

# 離散化させなくてはならない？(6分割**5変数の状態が生み出される)
# 状態s一覧
#
# self.m_dX
# self.m_dY
# self.m_dNeck
# self.m_dBallX
# self.m_dBallY
#
# 行動a一覧
# (turn 0)
# (turn 60)
# (turn -60)
# (dash 100)
# (dash -100)
# (kick 100 0)
# (kick 50 0)
