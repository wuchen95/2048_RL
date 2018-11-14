#-*-coding:utf-8-*-
from RL_brain import DeepQNetwork
from game2048 import UI


def run_maze():
    step = 0    # 用来控制什么时候学习
    for episode in range(200000):
        to_show = False

        if(episode%2000 == 0): # 每500轮展示一次
            to_show = True
        print(episode, "score: ", env.game.score)
        # 初始化环境
        observation = env.reset(to_show)


        while True:
            # 刷新环境
            if to_show:
                env.root.update()

            # DQN 根据观测值选择行为
            action = RL.choose_action(observation)

            # 环境根据行为给出下一个 state, reward, 是否终止
            observation_, reward, done = env.step(action, to_show)

            # DQN 存储记忆
            RL.store_transition(observation, action, reward, observation_)

            # 控制学习起始时间和频率（先累计一些记忆再开始学习） 每五轮进行一次学习
            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # 讲下一个 state_ 变为 下次循环的 state
            observation = observation_


            # 如果终止，就跳出循环
            if done:
                break
            step +=1  # 总步数
    # end of game
    print('game over')


if __name__ == "__main__":
    env = UI()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=1000,    # 每 200 步替换一次 target_net 的参数
                      memory_size=20000   # 记忆上限
                      # output_graph=True  # 是否输出 tensorboard 文件
                      )
    env.root.after(0, run_maze)   # 100为 100ms sleep
    env.root.mainloop()
    RL.plot_cost()   # 观看神经网络的误差曲线