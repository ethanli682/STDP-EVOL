import argparse
import time

from bindsnet.network.topology_features import Weight
import numpy as np
import torch
import matplotlib

from bindsnet.analysis.plotting import plot_weights

import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from bindsnet.analysis.plotting import plot_spikes  # plot_performance

# from bindsnet.encoding import bernoulli
from bindsnet.encoding import poisson
from bindsnet.environment.dot_simulator import DotSimulator

# from bindsnet.pipeline import EnvironmentPipeline
# from bindsnet.learning import MSTDP
from bindsnet.learning import MSTDPET, PostPre
from bindsnet.network import Network

# from bindsnet.pipeline.action import select_softmax
# from bindsnet.network.nodes import AbstractInput
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.topology import MulticompartmentConnection

# Handle arguments for dot tracing params.
parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--dim", type=int, default=28)
parser.add_argument("--granularity", type=int, default=100)
parser.add_argument("--neurons", type=int, default=100)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--trn_eps", type=int, default=600)
parser.add_argument("--tst_eps", type=int, default=50)
parser.add_argument("--decay", type=int, default=4)
parser.add_argument("--herrs", type=int, default=0)
parser.add_argument("--diag", type=bool, default=False)
parser.add_argument("--randr", type=float, default=0.15)
parser.add_argument("--boundh", type=str, default="bounce")
parser.add_argument("--fit_func", type=str, default="dir")
parser.add_argument("--allow_stay", type=bool, default=False)
parser.add_argument("--pandas", type=bool, default=False)
parser.add_argument("--mute", type=bool, default=False)
parser.add_argument("--write", type=bool, default=False)
parser.add_argument("--fcycle", type=int, default=100)
parser.add_argument("--gpu", type=bool, default=True)
parser.add_argument("--sleep", type=bool, default=False)

args = parser.parse_args()

steps = args.steps  # timesteps in which the dot is moving
dim = args.dim  # 28x28 square
granularity = args.granularity  # granularity (or precision) of spike trains
neurons = args.neurons  # Number of neurons in hidden layer
dt = args.dt  # delta time of network
trn_eps = args.trn_eps  # training episodes
tst_eps = args.tst_eps  # testing episodes
decay = args.decay  # length of decaing tail behind a dot
herrs = args.herrs  # distraction dots
diag = args.diag  # allows diagonal movement
randr = args.randr  # determines rate of randomization of movement
boundh = args.boundh  # bounds handling mode
fit_func = args.fit_func  # fitness function
allow_stay = args.allow_stay  # disable option for targets to remain in place.
pandas = args.pandas  # true = pandas DF printout; false = heatmap
mute = args.mute  # prohibit graphical rendering
write = args.write  # write observed grids to file.
fcycle = args.fcycle  # number of episodes per save file
gpu = args.gpu  # Utilize cuda
sleep = args.sleep  # sleep or not to sleep 


if diag:
    moveChoices = 9
else:
    moveChoices = 5

""" Set some globals """
# Set processor type
if torch.cuda.is_available() and gpu:
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Set neural network layer names
LAYER1 = "Input"
LAYER2 = "Hidden"
LAYER3 = "Output"

# file path for recording grid observations, rewards, and performance.
OUT_FILE_PATH = "out/"


def genFileName(ftype, suffix=""):
    """
    Generates output file names for rewards and performance
    """
    # Grab system time and trim off extra large parts of the number.
    sysTime = time.time()
    sysTime = int(1e10 * (sysTime - 1e6 * (sysTime // 1e6)))

    # Create filename if one isn't provided.
    return OUT_FILE_PATH + ftype + "_s" + str(sysTime) + "_" + suffix + ".csv"


def runSimulator(net, env, spikes, episodes, gran=100, rfname="", pfname=""):
    steps = env.timesteps
    dt = net.dt
    spike_ims, spike_axes = None, None

    # SLEEP IMPLEMENTATION PARAMS
    SLEEP_INTERVAL = 10      
    LEAP_THRESHOLD = 1.3     
    LEAP_BONUS = 5.0         
    
    # reward increasing params 
    MAX_MULTIPLIER = 5.0     # rewards are stronger toward beginning of training
    DECAY_RATE = 5.0 / episodes # reward decays as episodes reaches the end
    # =======================


    # For each episode...
    for ep in range(episodes):
        # Reset variables for new episode.
        total_reward = 0
        rewards = np.zeros(steps)
        intercepts = 0
        step = 0
        net.reset_state_variables()
        env.reset()
        env.dotDir = 0    # ADDED TO PAUSE THE DOT
        env.randr = -1.0  # ADDED TO PREVENT RANDOM MOVEMENT
        done = False
        env.render()
        clock = time.time()

        # === ADD THESE INITIALIZERS FOR THE WEIGHT PLOTS ===
        im_in_hid = None
        im_recur = None
        im_hid_out = None
        # ===================================================


        is_sleeping = (ep % SLEEP_INTERVAL == 0) and ep > 0

        if is_sleeping:
            print(f"--- Episode {ep}: Initiating Sleep/Consolidation Phase ---")

        # multiplier for the reward
        # anneal_mult = 1.0 + (MAX_MULTIPLIER - 1.0) * np.exp(-DECAY_RATE * ep)

        # Initialize action tensor, network output monitor, and spike train record.
        action = torch.randint(low=0, high=env.action_space.n, size=(1,))[0]
        spike_record = torch.zeros(
            (steps, int(gran / dt), env.action_space.n), device=DEVICE
        )
        # perf_ax = None

        # Run through episode.
        while not done:
            # Run through episode.
            step += 1

            
            # SLEEP: During sleep episodes, the agent receives no meaningful sensory 
            # input and no reward signal, but the network still processes "noise" 
            # and updates its weights based on the last received reward. This 
            # allows the network to consolidate learning from previous episodes 
            # without new external stimuli.
            if is_sleeping and sleep:
                # 1. Sensory deprivation: random background noise instead of actual vision
                obs = torch.rand((dim, dim)) * 0.1  # Low-level static
                obs = obs.to(DEVICE)
                
                # 2. No dopamine during sleep
                reward = torch.tensor(0.0, dtype=torch.float32).to(DEVICE)
                
                # We must still step the environment to keep the clock moving
                _, _, done, _ = env.step(action) 
            
            else:
                # 1. Calculate distance BEFORE the step
                prev_dist = np.sqrt((env.netDot.row[0] - env.dots[0].row[0])**2 + 
                                    (env.netDot.col[0] - env.dots[0].col[0])**2)

                # 2. Take the step
                obs, reward, done, intercept = env.step(action)
                obs = torch.Tensor(obs).to(DEVICE)
                
                # 3. Calculate distance AFTER the step
                curr_dist = np.sqrt((env.netDot.row[0] - env.dots[0].row[0])**2 + 
                                    (env.netDot.col[0] - env.dots[0].col[0])**2)

                # 4. Calculate the gradient reward
                custom_reward = prev_dist - curr_dist

                # 5. Add the "Bullseye" multiplier for actual interceptions
                if intercept:
                    custom_reward += 20.0  
                    
                # PUNISHMENT_MULTIPLIER = 3.0  # Adjust this to tune how harshly it learns from mistakes
                # if custom_reward < 0:
                #     custom_reward *= PUNISHMENT_MULTIPLIER

                # The Outsized Leap Reward 
                if custom_reward >= LEAP_THRESHOLD:
                    custom_reward += LEAP_BONUS

                # custom_reward *= anneal_mult

                # 6. Overwrite the environment's default reward
                reward = torch.tensor(custom_reward, dtype=torch.float32).to(DEVICE)
                # Determine the action probabilities
                probabilities = torch.softmax(
                    torch.sum(spike_record[step - 1 % steps], dim=0), dim=0
                )

                action = torch.multinomial(probabilities, num_samples=1).item()

                # Place the observations into the inputs.
                obs = obs.unsqueeze(0)
                inputs = {LAYER1: poisson(obs * 5e2, gran, dt, device=DEVICE)}
                if DEVICE == "cuda":
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                # Run the network on the spike train-encoded inputs.
                net.run(inputs=inputs, time=gran, reward=reward)
                spike_record[step % steps] = spikes[LAYER3].get("s").squeeze()
                rewards[step - 1] = reward.item()

                # record successful intercept
                if intercept:
                    intercepts += 1

                if done:
                    # Update network with cumulative reward
                    if net.reward_fn is not None:
                        net.reward_fn.update(accumulated_reward=total_reward, steps=step)

                    # Save rewards thus far to file
                    if rfname != "":
                        with open(rfname, "ab") as f:
                            np.savetxt(f, rewards, delimiter=",", fmt="%.6f")

                spikes_ = {layer: spikes[layer].get("s").view(gran, -1) for layer in spikes}
                spike_ims, spike_axes = plot_spikes(spikes_, ims=spike_ims, axes=spike_axes)
                # perf_ax = plot_performance(reward, x_scale=10, ax=perf_ax)

                env.render()
                total_reward += reward

            if step % 10 == 0:
                print(
                    f"Iteration: {step} (Time: {time.time() - clock:.4f}); reward: {reward}"
                )
                clock = time.time()
        print(f"Episode {ep} total reward:{total_reward}")
        # # === ADD THIS BLOCK TO DISPLAY AND UPDATE WEIGHTS ===
        # # Pass the raw tensors; plotting.py handles the detachment automatically
        # w_in_hid = net.connections[(LAYER1, LAYER2)].w
        # w_recur = net.connections[(LAYER2, LAYER2)].w
        # w_hid_out = net.connections[(LAYER2, LAYER3)].w

        # im_in_hid = plot_weights(w_in_hid, wmin=0, wmax=1, im=im_in_hid, title="Input -> Hidden", figsize=(4, 4))
        # im_recur = plot_weights(w_recur, wmin=0, wmax=1, im=im_recur, title="Recurrent", figsize=(4, 4))
        # im_hid_out = plot_weights(w_hid_out, wmin=0, wmax=1, im=im_hid_out, title="Hidden -> Output", figsize=(4, 4))

        # plt.pause(0.01) # Allows Matplotlib's GUI event loop to redraw the windows
        # # ====================================================
        # Save intcercepts thus far to file
        if pfname != "":
            with open(pfname, "a+") as f:
                if 0 < ep:
                    f.write("," + str(intercepts))
                else:
                    f.write(str(intercepts))

        # Cycle output files every 10000 iterations
        if ep % fcycle == 0:
            env.cycleOutFiles()



def main():
    # Build network.
    network = Network(dt=dt)

    # Input Layer
    inpt = Input(n=dim * dim, shape=[1, 1, 1, dim, dim], traces=True)

    # Hidden Layer
    middle = LIFNodes(n=neurons, traces=True, thresh=-35.0)

    # Ouput Layer
    out = LIFNodes(n=moveChoices, 
                   refrac=0, 
                   traces=True,
                   thresh=-35.0,  
                   )

    # # Connections from input layer to hidden layer
    # inpt_middle = Connection(source=inpt, target=middle, wmin=0, wmax=1)

    # # Connections from hidden layer to output layer
    # middle_out = Connection(
    #     source=middle,
    #     target=out,
    #     wmin=0,  # minimum weight value
    #     wmax=1,  # maximum weight value
    #     update_rule=MSTDPET,  # learning rule
    #     nu=1e-1,  # learning rate
    #     norm=0.5 * middle.n,  # normalization
    # )

    # # Recurrent connection, retaining data within the hidden layer
    # # CHANGE: dont learn for now
    # recurrent = Connection(
    #     source=middle,
    #     target=middle,
    #     wmin=0,  # minimum weight value
    #     wmax=1,  # maximum weight value
    #     # update_rule=PostPre,  # learning rule
    #     # nu=1e-1,  # learning rate
    #     norm=5e-3 * middle.n,  # normalization
    # )
    network.add_layer(inpt, name=LAYER1)
    network.add_layer(middle, name=LAYER2)
    network.add_layer(out, name=LAYER3)
# Get the exact number of neurons from your layers
    n_in = network.layers[LAYER1].n   # 784
    n_hid = network.layers[LAYER2].n  # 100
    n_out = network.layers[LAYER3].n  # 5

    # Create random weights, but multiply by 0.1 so they start weak!
    w_in_hid_init = torch.rand(n_in, n_hid) * 0.1
    w_recur_init = torch.rand(n_hid, n_hid) * 2 - 1  # Values between -1 and 1
    mask_recur = torch.rand(n_hid, n_hid) < 0.2  # 20% negative, 80% positive
    w_recur_init = torch.where(mask_recur, -torch.abs(w_recur_init), torch.abs(w_recur_init))
    w_hid_out_init = torch.rand(n_hid, n_out) * 0.1

    # 1. Input -> Hidden

    # weight_feature = Weight(name="input_weight_feature", value=w_in_hid_init)

    in_hid = MulticompartmentConnection(
        source=network.layers[LAYER1],
        target=network.layers[LAYER2],
        w=w_in_hid_init,             # Pass the scaled weights here
        device=DEVICE,
        wmin=0.0,
        wmax=1.0
    )
    network.add_connection(in_hid, source=LAYER1, target=LAYER2)

    # 2. Recurrent (Hidden -> Hidden)
    recur = MulticompartmentConnection(
        source=network.layers[LAYER2],
        target=network.layers[LAYER2],
        w=w_recur_init,              # Pass the scaled recurrent weights here
        device=DEVICE, 
        wmin=0.0,
        wmax=1.0
    )
    network.add_connection(recur, source=LAYER2, target=LAYER2)

    # 3. Hidden -> Output
    hid_out = MulticompartmentConnection(
        source=network.layers[LAYER2],
        target=network.layers[LAYER3],
        w=w_hid_out_init,            # Pass the scaled output weights here
        update_rule=MSTDPET,
        device=DEVICE,
        nu=1e-2,   
        wmin=0.0,
        wmax=1.0
    )
    network.add_connection(hid_out, source=LAYER2, target=LAYER3)    # Add all layers and connections to the network.

    # network.add_connection(inpt_middle, source=LAYER1, target=LAYER2)
    # network.add_connection(middle_out, source=LAYER2, target=LAYER3)
    # network.add_connection(recurrent, source=LAYER2, target=LAYER2)
    network.to(DEVICE)

    # Add monitors
    # network.add_monitor(Monitor(network.layers["Hidden"], ["s"], time=granularity), "Hidden")
    # network.add_monitor(Monitor(network.layers["Output"], ["s"], time=granularity), "Output")
    spikes = {}
    for layer in set(network.layers):
        spikes[layer] = Monitor(
            network.layers[layer],
            state_vars=["s"],
            time=int(granularity / dt),
            device=DEVICE,
        )
        network.add_monitor(spikes[layer], name=layer)

    # Load the Dot Simultation environment.
    environment = DotSimulator(
        steps,
        decay=decay,
        herrs=herrs,
        diag=diag,
        randr=randr,
        write=write,
        mute=mute,
        bound_hand=boundh,
        fit_func=fit_func,
        allow_stay=allow_stay,
        pandas=pandas,
        fpath=OUT_FILE_PATH,
    )
    environment.reset()

    print("Training: ")
    rewFile = genFileName("rew", "train")
    perfFile = genFileName("perf", "train")
    environment.addFileSuffix("train")
    runSimulator(
        network,
        environment,
        spikes,
        episodes=trn_eps,
        gran=granularity,
        rfname=rewFile,
        pfname=perfFile,
    )

    # Freeze learning
    network.learning = False

    print("Testing: ")
    rewFile = genFileName("rew", "test")
    perfFile = genFileName("perf", "test")
    environment.changeFileSuffix("train", "test")
    runSimulator(
        network,
        environment,
        spikes,
        episodes=tst_eps,
        gran=granularity,
        rfname=rewFile,
        pfname=perfFile,
    )


if __name__ == "__main__":
    main()
