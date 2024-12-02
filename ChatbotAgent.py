import gym
from gym import spaces
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import DummyVecEnv


class HelpDeskChatbotEnv(gym.Env):
    def __init__(self):
        super(HelpDeskChatbotEnv, self).__init__()

        # Define action space:
        # 0 - Use Model 1 (optimistic, specific answers)
        # 1 - Use Model 2 (default, neutral prompt)
        # 2 - Use Model 3 (advanced, detailed ideas)
        # 3 - Pass to agent
        # 4 - Give Discount
        self.action_space = spaces.Discrete(5)

        # Define state space:
        # Sentiment (0 - negative, 1 - positive)
        # Number of replies (normalized: 0 - few, 1 - many)
        # Time between replies (normalized to [0, 1])
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)

        # Initialize state
        self.state = None
        self.reset()

    def reset(self):
        # Random initial state
        self.state = np.array([
            np.random.uniform(0.0, 1.0),  # Sentiment: Full range
            np.random.uniform(0.0, 1.0),  # Replies: Few initial replies
            np.random.uniform(0.0, 1.0)   # Time between replies: Partial range
        ])
        return self.state

    def step(self, action):
        sentiment, replies, time_between_replies = self.state

        # Simulate action impact
        if action == 0:  # Use Model 1 encouraging
            sentiment += np.random.uniform(0.1, 0.2)
            time_between_replies += np.random.uniform(0.1, 0.2)
            replies += 0.1
        elif action == 1:  # Use Model 2 default
            sentiment += np.random.uniform(-0.1, 0.2)
            time_between_replies += np.random.uniform(-0.1, 0.3)
            replies += 0.1
        elif action == 2:  # Use Model 3 advanced
            sentiment += np.random.uniform(0.1, 0.4)
            time_between_replies += np.random.uniform(-0.1, 0.3)
            replies += 0.3
        elif action == 3:  # Pass to Agent
            sentiment += np.random.uniform(0.1, 0.5)
            replies = 1.0  # Mark query as resolved
            time_between_replies += 0.3
        elif action == 4:  # Give Discount
            sentiment += np.random.uniform(-0.1, 0.4)  # Smaller sentiment boost
            time_between_replies += 0.3  # Higher penalty for this action
            replies += 0.4  # Slightly slower resolution

        # Update replies
        replies = min(1.0, replies)

        # Calculate reward
        query_complete = replies >= 1.0

        reward = sentiment - time_between_replies  # Base reward

        # Bonuses
        if query_complete and action != 3:  # Resolving autonomously
            reward += 2
        if 0.5 <= sentiment :  # Moderate sentiment bonus
            reward += 0.5
        if time_between_replies < 0.3:  # Quick replies
            reward += 0.5
            
        if action == 0 and sentiment < 0.5 and time_between_replies < 0.4:  # Reward Model 0 for optimistic
            reward += 2
        if action == 1 and sentiment < 0.7 and time_between_replies > 0.3:  # Reward Model 1 as default
            reward += 1
        if action == 2 and sentiment > 0.7 and time_between_replies < 0.3:  # Reward Model 2 as advanced
            reward += 0.5
        if action == 3 and sentiment < 0.4 and time_between_replies > 0.6:  # Reward passing to agent strategically
            reward +=1.5
        if action == 4 and sentiment > 0.6 and time_between_replies < 0.5:  # Reward Model 4 discount
            reward += 1

        
        
        
        # Penalties
        if sentiment < 0.5:  # Penalize bad sentiment
            reward -= 2
        #if action == 4 and sentiment > 0.6:  # Penalize unnecessary discounts
            #reward -= 0.2
        if time_between_replies > 0.5:  # Penalize high delays
            reward -= 1


        # Clamp values
        sentiment = max(0, min(1, sentiment))
        time_between_replies = max(0, min(1, time_between_replies))
        reward = max(-15, min(reward, 15))  # Scale reward to [-1, 1]

        # Update state
        self.state = np.array([sentiment, replies, time_between_replies])

        # Check if episode is done
        done = query_complete

        return self.state, reward, done, {}

# Initialize the environment
env = HelpDeskChatbotEnv()


if __name__ == "__main__":    
        # Wrap environment for SB3 compatibility
    env = DummyVecEnv([lambda: HelpDeskChatbotEnv()])

    # Initialize DQN model
    #lower gamma - short term rewards larger .9 gamma is long term rewards
    model_chatbot = DQN('MlpPolicy', env, verbose=1, learning_rate=0.01, gamma=0.2, exploration_fraction=0.4, exploration_final_eps=0.01)


    rewardlist_chat=[]

    # Train model with a callback to track rewards
    def customcallback(locals_, globals_):
        rewardlist_chat.append(locals_['rewards'])  # Append the reward of the current step to the list
        return True  # Always return True to continue training

    # Train the model
    model_chatbot.learn(total_timesteps=10000, callback= customcallback)
    
    
    import matplotlib.pyplot as plt
    # Smoothing with moving average
    window_size = 100  # Adjust the window size as needed
    rew= [i[0] for i in rewardlist_chat]
    smoothed_rewards = np.convolve(rew, np.ones(window_size)/window_size, mode='valid')

    # Plot the reward trend
    plt.figure(figsize=(8,4))
    plt.plot(smoothed_rewards)

    plt.xlabel('Timesteps')
    plt.ylabel('Reward')
    plt.grid()
    plt.title('Training Reward Trend')
    plt.show()


    from stable_baselines3 import DQN
    import numpy as np
    # Save and load the trained model
    model_chatbot.save("chatbot_agent_dqn")