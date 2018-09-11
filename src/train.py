import subprocess
import os
import time
from memory import Memory
import pickle
from actor import Actor
from critic import Critic

# エピソード数
episodes = 100000
# 実行ファイル名
exe_program = "james.py"
# 状態空間次元数
state_num = 6
# 行動空間次元数
action_num = 1

if __name__ == "__main__":
    # ログが残っていればその続きのエピソードとして学習を開始する。
    start_episode = 0
    if os.path.isfile("./logs/jamesleft_1_reward.log"):
        with open("./logs/jamesleft_1_reward.log", "r") as file:
            lines = file.readlines()
            last_episode = lines[-1].split(",")[0]
            start_episode = last_episode

    # actor critic memoryのインスタンスの作成
    actor = Actor(action_num, state_num)
    critic = Critic(action_num, state_num, actor.sess)
    memory = Memory()

    # ./actor_and_criticの作成
    if not os.path.isdir("./A_C_M"):
        os.mkdir("./A_C_M")

    # actor critic memoryのインスタンスの保存
    actor_file = open("./A_C_M/actor_file", "wb")
    critic_file = open("./A_C_M/critic_file", "wb")
    memory_file = open("./A_C_M/memory_file", "wb")
    pickle.dump(actor, actor_file, protocol=2)
    pickle.dump(critic, critic_file, protocol=2)
    pickle.dump(memory, memory_file, protocol=2)
    actor_file.close()
    critic_file.close()
    memory_file.close()

    # pretrain(100step分だけ回す)=============================
    print("pretrain start!!")
    # ディレクトリの移動
    os.chdir("../")
    os.chdir("../")
    # サーバの起動
    cmd = \
        "rcssserver server::half_time = -1 server::send_step = 3 server::sense_body_step = 2 server::simulator_step = 2 server::auto_mode = true server::kick_off_wait = 200"
    server = subprocess.Popen(cmd.split())
    # モニタの起動
    cmd = "soccerwindow2"
    window = subprocess.Popen(cmd.split())
    # ディレクトリの移動
    os.chdir("./James/src")
    # クライアントプログラムの実行
    cmd = "python3 {} {}".format(exe_program, "pretrain")
    cliant = subprocess.Popen(cmd.split())
    # pretrainの終了まで待機
    time.sleep(10)
    print("pretrain end!!")
    # ========================================================



    # 本番学習スタート
    print("start")
    for episode in range(int(start_episode), episodes):
        # ディレクトリの移動
        os.chdir("../")
        os.chdir("../")

        # サーバの起動
        cmd = \
            "rcssserver server::half_time = -1 server::send_step = 3 server::sense_body_step = 2 server::simulator_step = 2 server::auto_mode = true server::kick_off_wait = 200"
        server = subprocess.Popen(cmd.split())

        # モニタの起動
        cmd = "soccerwindow2"
        window = subprocess.Popen(cmd.split())

        # ディレクトリの移動
        os.chdir("./James/src")

        if not os.path.isdir("./npy"):
            os.mkdir("./npy")

        if not os.path.isdir("./logs"):
            os.mkdir("./logs")

        # クライアントプログラムの実行
        cmd = "python3 {} {} {}".format(exe_program, "train", episode)
        cliant = subprocess.Popen(cmd.split())

        # 学習
        # while True:
        #     if zidan2.episode_finish_flag is True:
        #         break


        time.sleep(10)


        print("episode{} is done ".format(episode))

        # サーバの削除
        server.kill()
        # ウィンドウの削除
        window.kill()
        # クライアントの削除
        cliant.kill()

    print("end")
