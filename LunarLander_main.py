import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt

class ReplayBuffer():
    def __init__(self,max_size,input_dims):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *input_dims),dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype= np.float32)
        self.action_memory = np.zeros(self.mem_size,dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int32)

    def store_trasition(self,state,action,reward,state_,done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = 1-int(done)
        self.mem_cntr +=1

    def sample_buffer(self,batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace= False)
        state = self.state_memory[batch]
        state_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal  = self.terminal_memory[batch]

        return state, actions, rewards, state_, terminal

def build_dqn (lr, n_actions, input_dims, fc1_dims, fc2_dims):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(fc1_dims,activation='relu'),
        tf.keras.layers.Dense(fc2_dims,activation='relu'),
        tf.keras.layers.Dense(n_actions,activation= None)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),loss= tf.keras.losses.mean_squared_error)

    return  model

class Agent():
    def __init__(self, lr,gamma,n_actions, epsilon,batch_size, input_dims, epsilon_dec =1e-3, epsilon_end=0.01,
                 mem_size = 1000000,fname = 'project1_youtube.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.gamma =gamma
        self.epsilon = epsilon
        self.eps_min = epsilon_end
        self.eps_dec = epsilon_dec
        self.batch_size = batch_size
        self.model_file = fname
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = build_dqn(lr,n_actions,input_dims,256,256)

    def store_transition(self, state, action, reward, new_state,done):
        self.memory.store_trasition(state,action, reward,new_state,done)

    def choose_action (self, observation):
        if np.random.random()<self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            actions = self.q_eval.predict(state)

            action = np.argmax(actions)
        return action
    def learn (self):
        if self.memory.mem_cntr<self.batch_size:
            return

        states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)
        q_eval = self.q_eval.predict(states)
        q_next = self.q_eval.predict(states_)

        q_target = np.copy(q_eval)
        batch_index = np.arange(self.batch_size, dtype = np.int32)

        q_target[batch_index,actions] = rewards + self.gamma*np.max(q_next,axis=1)*dones
        self.q_eval.train_on_batch(states,q_target)
        self.epsilon =self.epsilon-self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = tf.keras.models.load_model(self.model_file)



if __name__=='__main__':
    tf.compat.v1.disable_eager_execution()
    env = gym.make('LunarLander-v2')
    env.reset()
    lr = 0.001
    n_games = 500
    agent = Agent(lr =lr, gamma=0.95,epsilon=1.0, input_dims=env.observation_space.shape,
                  n_actions=env.action_space.n,mem_size=1000000, batch_size=64,epsilon_end=0.01)
    scores = []
    eps_history = []

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_,reward,done,info = env.step(action)
            score+=reward
            agent.store_transition(observation,action,reward,observation_,done)
            observation=observation_
            agent.learn()

        eps_history.append(agent.epsilon)
        scores.append(score)
        avg_scores = np.mean(scores[-100:])
        print('episode: ',i, 'score %.2f' % score, 'average_sacore %.2f' % avg_scores, 'epsilon %.2f' % agent.epsilon)


    plt.plot(eps_history,scores)
    plt.title('Score in episode')
    plt.show()