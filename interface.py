from rl_env import FlexSimEnv, STATE_KEYS

env = FlexSimEnv()
obs, info = env.reset()
print("-------------")
for key in obs:
    print(f"{key}: {obs[key]}")
print(f'info: {info}')

## for every step, print out the state, action, request input from user for action
## if they enter exit then terminate the loop
terminated = False
while not terminated or env.env.timestamps[-1] < 4000:
    print("-------------")
    action = input("Enter action: ")
    if action == 'exit':
        terminated = True
    else:
        action = int(action)
        obs, reward, terminated, truncated, info = env.step(action)
        print("-------------")
        print(f"Reward: {reward}, Info: {info}")
        for key in obs:
            print(f"{key}: {obs[key]}")
## behaving reasonably by time 4951 i am getting cumulative reward of -4
