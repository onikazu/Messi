import subprocess
import os
import time

# エピソード数
episodes = 100000
# 実行ファイル名
exe_program = "messi.py"
# 状態空間次元数
state_num = 4
# 行動数
action_num = 5

if __name__ == "__main__":
    # ログが残っていればその続きのエピソードとして学習を開始する。
    start_episode = 0
    if os.path.isfile("./logs/jamesleft_1_reward.log"):
        with open("./logs/jamesleft_1_reward.log", "r") as file:
            lines = file.readlines()
            last_episode = lines[-1].split(",")[0]
            start_episode = last_episode

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
        os.chdir("./Messi/src")

        if not os.path.isdir("./models"):
            os.mkdir("./models")

        if not os.path.isdir("./logs"):
            os.mkdir("./logs")

        # クライアントプログラムの実行
        cmd = "python3 {} {}".format(exe_program, episode)
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
