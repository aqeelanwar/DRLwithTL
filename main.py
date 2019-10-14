# Branch - DFA Implementation
import sys
from network.agent import DeepAgent
from environments.initial_positions import *
import os
import psutil
from os import getpid
from network.Memory import Memory
from aux_functions import *
from configs.read_cfg import read_cfg

# Debug message suppressed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

process = psutil.Process(getpid())

cfg = read_cfg(config_filename='configs/config.cfg', verbose=True)


input_size = 227
num_actions = cfg.num_actions

screen = pygame_connect(H=925, W=380)

dropout_rate = cfg.dropout
env_type=cfg.env_type
env_name = cfg.env_name
train_type = cfg.train_type # 'e2e', 'last4', 'last3', 'last2'
weights_type = 'Imagenet'

reset_array, level_name, crash_threshold, initZ = initial_positions(env_name)

epsilon_annealing = cfg.epsilon_saturation
wait_before_train = cfg.wait_before_train
train_interval=cfg.train_interval
max_iters = cfg.max_iters
gamma = cfg.gamma
update_target = cfg.update_target_interval
buffer_length = cfg.buffer_len
ReplayMemory = Memory(cfg.buffer_len)
switch_env_steps = cfg.switch_env_steps
batch_size=cfg.batch_size
Q_clip=cfg.Q_clip
custom_load=cfg.custom_load
custom_load_path = cfg.custom_load_path
lr = cfg.lr
epsilon = cfg.epsilon
epsilon_model = cfg.epsilon_model

# Save the network to the directory network_path
if custom_load == True:
    network_path = 'models/trained/' + env_type + '/' + env_name + '/' + 'CustomLoad/' + train_type + '/'+ train_type
else:
    network_path = 'models/trained/' + '/' + env_type + '/' + env_name + '/' + weights_type + '/' + train_type + '/'+ train_type

if not os.path.exists(network_path):
    os.makedirs(network_path)

# Connect to Unreal Engine and get the drone handle: client
client, old_posit = connect_drone()

# Define DQN agents
agent = DeepAgent(input_size, num_actions, client, env_type,train_type,network_path, name='DQN')
target_agent = DeepAgent(input_size, num_actions, client, env_type,train_type, network_path, name='Target')

# Load custom weights from custom_load_path if required
if custom_load==True:
    print('Loading weights from: ', custom_load_path)
    agent.load_network(custom_load_path)
    target_agent.load_network(custom_load_path)



iter = 0
num_col1 = 0
epi1 = 0

active = True
action_type = 'Wait_for_expert'

automate = True
epsilon_greedy = True
choose=False
print_qval=False
last_crash1=0
environ=True
e1  =0
e2 = 0
ret = 0
dist = 0
switch_env=False

save_posit = old_posit

level_state = [None]*len(level_name)
level_posit = [None]*len(level_name)
last_crash_array = np.zeros(shape=len(level_name), dtype=np.int32)
ret_array = np.zeros(shape=len(level_name))
dist_array = np.zeros(shape=len(level_name))
epi_env_array = np.zeros(shape=len(level_name), dtype=np.int32)

level = 0
times_switch = 0
curr_state1 = agent.get_state()

i = 0
log_path = network_path+'log.txt'
f = open(log_path, 'w')


while active:
    try:

        active, automate, lr, client = check_user_input(active, automate, lr, epsilon, agent, network_path, client, old_posit, initZ)

        if automate:
            start_time = time.time()
            if switch_env:
                posit1_old = client.simGetVehiclePose()
                times_switch=times_switch+1
                level_state[level] = curr_state1
                level_posit[level] = posit1_old
                last_crash_array[level] = last_crash1
                ret_array[level] = ret
                dist_array[level] = dist
                epi_env_array[int(level/3)] = epi1

                level = (level + 1) % len(reset_array)

                print('Transferring to level: ', level ,' - ', level_name[level])

                if times_switch < len(reset_array):
                    reset_to_initial(level, reset_array, client)
                else:
                    curr_state1 = level_state[level]
                    posit1_old = level_posit[level]

                    reset_to_initial(level, reset_array, client)
                    client.simSetVehiclePose(posit1_old, ignore_collison=True)
                    time.sleep(0.1)

                last_crash1 = last_crash_array[level]
                ret = ret_array[level]
                dist = dist_array[level]
                epi1 = epi_env_array[int(level/3)]
                xxx = client.simGetVehiclePose()
                environ = environ^True


            action1, action_type1, epsilon, qvals = policy(epsilon, curr_state1, iter, epsilon_annealing, epsilon_model,  wait_before_train, num_actions, agent)

            action_word1 = translate_action(action1, num_actions)
            # Take the action
            agent.take_action(action1, num_actions)
            time.sleep(0.05)

            posit = client.simGetVehiclePose()

            new_state1 = agent.get_state()
            new_depth1, thresh = agent.get_depth()

            # Get GPS information
            posit = client.simGetVehiclePose()
            orientation = posit.orientation
            position = posit.position
            old_p = np.array([old_posit.position.x_val, old_posit.position.y_val])
            new_p = np.array([position.x_val, position.y_val])
            # calculate distance
            dist = dist + np.linalg.norm(new_p - old_p)
            old_posit = posit

            reward1, crash1 = agent.reward_gen(new_depth1, action1, crash_threshold, thresh)

            ret = ret+reward1
            agent_state1 = agent.GetAgentState()

            if agent_state1.has_collided:
                # if car_state.collision.object_id==77:

                num_col1 = num_col1 + 1
                print('crash')
                crash1 = True
                reward1 = -1
            data_tuple=[]
            data_tuple.append([curr_state1, action1, new_state1, reward1, crash1])
            err = get_errors(data_tuple, choose, ReplayMemory, input_size, agent, target_agent, gamma, Q_clip)
            ReplayMemory.add(err, data_tuple)

            # Train if sufficient frames have been stored
            if iter > wait_before_train:
                if iter%train_interval==0:
                # Train the RL network
                    old_states, Qvals, actions, err, idx = minibatch_double(data_tuple, batch_size, choose, ReplayMemory, input_size, agent, target_agent, gamma, Q_clip)

                    for i in range(batch_size):
                        ReplayMemory.update(idx[i], err[i])

                    if print_qval:
                        print(Qvals)

                    if choose:
                        # Double-DQN
                        target_agent.train_n(old_states, Qvals, actions, batch_size, dropout_rate, lr, epsilon, iter)
                    else:
                        agent.train_n(old_states, Qvals,actions,  batch_size, dropout_rate, lr, epsilon, iter)

                if iter % update_target == 0:
                    agent.take_action([-1], num_actions)
                    print('Switching Target Network')
                    choose = not choose
                    agent.save_network(network_path)

            iter += 1

            time_exec = time.time()-start_time
            VC = ''
            if environ:
                e1 = e1+1
                e_print=e1
            else:
                e2 = e2+1
                e_print = e2
            # init_p = epi1%len(init_pose_array)
            mem_percent = process.memory_info()[0]/2.**30

            s_log = 'Level :{:>2d}: Iter: {:>6d}/{:<5d} {:<8s}-{:>5s} Eps: {:<1.4f} lr: {:>1.8f} Ret = {:<+6.4f} Last Crash = {:<5d} t={:<1.3f} Mem = {:<5.4f}  Reward: {:<+1.4f}  '.format(
                    int(level),iter, epi1,
                    action_word1,
                    action_type1,
                    epsilon,
                    lr,
                    ret,
                    last_crash1,
                    time_exec,
                    mem_percent,
                    reward1)

            print(s_log)
            f.write(s_log+'\n')

            last_crash1=last_crash1+1

            if crash1:
                agent.return_plot(ret, epi1, int(level/3), mem_percent, iter, dist)
                ret=0
                dist=0
                epi1 = epi1 + 1
                last_crash1=0

                reset_to_initial(level, reset_array, client)
                time.sleep(0.2)
                curr_state1 =agent.get_state()
            else:
                curr_state1 = new_state1


            if iter%switch_env_steps==0:
                switch_env=True
            else:
                switch_env=False

            if iter% max_iters==0:
                automate=False

            # if iter >140:
            #     active=False



    except Exception as e:
        print('------------- Error -------------')
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(exc_obj)
        automate = False
        print('Hit r and then backspace to start from this point')





