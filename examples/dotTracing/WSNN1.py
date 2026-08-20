import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from bindsnet.analysis.plotting import plot_spikes  # plot_performance

# from bindsnet.encoding import bernoulli
from bindsnet.encoding import poisson
from bindsnet.environment.dot_simulator import DotSimulator

# from bindsnet.pipeline import EnvironmentPipeline
# from bindsnet.learning import MSTDP
from bindsnet.learning import MSTDPET, PostPre
from bindsnet.network import Network, network
from bindsnet.network.topology_features import Probability, Weight



# from bindsnet.pipeline.action import select_softmax
# from bindsnet.network.nodes import AbstractInput
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import AdaptiveLIFNodes, Input, LIFNodes
from bindsnet.network.topology import MulticompartmentConnection

# Handle arguments for dot tracing params.
parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--dim", type=int, default=28)
parser.add_argument("--granularity", type=int, default=100)
parser.add_argument("--neurons", type=int, default=100)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--trn_eps", type=int, default=1000)
parser.add_argument("--tst_eps", type=int, default=100)
parser.add_argument("--decay", type=int, default=4)
parser.add_argument("--herrs", type=int, default=0)
parser.add_argument("--diag", type=bool, default=False)
parser.add_argument("--randr", type=float, default=0.15)
parser.add_argument("--boundh", type=str, default="bounce")
parser.add_argument("--fit_func", type=str, default="dir")
parser.add_argument("--allow_stay", type=bool, default=False)
parser.add_argument("--pandas", type=bool, default=False)
parser.add_argument("--mute", type=bool, default=False)
parser.add_argument("--write", type=bool, default=True)
parser.add_argument("--fcycle", type=int, default=100)
parser.add_argument("--gpu", type=bool, default=True)

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

from bindsnet.learning import LearningRule




######################## CUSTOM LEARNING RULES FOR WEIGHTLESS SPIKING NEURAL NETWORKS ########################
class DelaySTDP(LearningRule):
    """
    Hebbian STDP learning rule that modifies transmission delays instead of weights.
    Used to implement Weightless Spiking Neural Networks (WSNN).
    """
    def __init__(self, connection, nu, tc_pre=20.0, tc_post=20.0, max_delay=32, **kwargs):
        
        # --- FIX: Inject dummy attributes so the base LearningRule doesn't crash ---
        if not hasattr(connection, 'wmin'):
            connection.wmin = 0.0
        if not hasattr(connection, 'wmax'):
            connection.wmax = 1.0
        # -------------------------------------------------------------------------
            
        super().__init__(connection=connection, nu=nu, **kwargs)
        
        self.tc_pre = tc_pre
        self.tc_post = tc_post
        self.max_delay = max_delay
        

    def update(self, **kwargs) -> None:
        pre_trace = self.connection.source.trace
        post_trace = self.connection.target.trace
        
        if pre_trace is None or post_trace is None or self.connection.delay is None:
            return

        # Calculate Delta t update
        delay_update = self.nu[0] * torch.ger(pre_trace.view(-1), post_trace.view(-1))
        
        # Myelination (highly correlated spikes decrease delay)
        new_delays = self.connection.delay.float() - delay_update
        
        # Demyelination (forgetting factor)
        new_delays += 0.00005
        
        # Enforce integer timestep bounds
        self.connection.delay = torch.clamp(new_delays, 0.0, self.max_delay).long()

class ModulatedDelaySTDP(LearningRule):
    """
    Reward-Modulated Hebbian STDP for Weightless Spiking Neural Networks.
    Delays are updated based on an Eligibility Trace multiplied by an external Reward.
    """
    def __init__(self, connection, nu, tc_pre=20.0, tc_post=20.0, tc_e=1000.0, max_delay=32, **kwargs):
        print("Updating ModulatedDelaySTDP...")
        # Inject dummy attributes so the base LearningRule doesn't crash
        if not hasattr(connection, 'wmin'):
            connection.wmin = 0.0
        if not hasattr(connection, 'wmax'):
            connection.wmax = 1.0
            
        super().__init__(connection=connection, nu=nu, **kwargs)
        
        self.tc_pre = tc_pre
        self.tc_post = tc_post
        self.tc_e = tc_e           # Time constant for eligibility trace decay
        self.max_delay = max_delay
        
        # Initialize the eligibility trace to match the shape and device of the delay matrix
        self.eligibility_trace = torch.zeros_like(self.connection.delay, dtype=torch.float)

    def update(self, **kwargs) -> None:
        print("Updating ModulatedDelaySTDP...")
        pre_trace = self.connection.source.trace
        post_trace = self.connection.target.trace
        
        if pre_trace is None or post_trace is None or self.connection.delay is None:
            return

        # 1. Calculate standard STDP overlap (Hebbian coincidence)
        stdp_overlap = self.nu[0] * torch.ger(pre_trace.view(-1), post_trace.view(-1))
        
        # 2. Update the Eligibility Trace
        # Decay the old trace slightly and add the new spike correlations
        decay_factor = 1.0 - (1.0 / self.tc_e)
        self.eligibility_trace = (self.eligibility_trace * decay_factor) + stdp_overlap
        
        # 3. Extract the Reward
        # BindsNET automatically passes the reward from network.run() into **kwargs
        reward = kwargs.get("reward", 0.0)
        
        # Ensure reward is a scalar float
        if isinstance(reward, torch.Tensor):
            reward = reward.item()

        # 4. Calculate the Delay Update (Reward * Trace)
        delay_update = reward * self.eligibility_trace
        
        # 5. Apply the update
        # If reward is POSITIVE (closer to dot), delay_update is positive.
        # Subtracting it DECREASES the delay (myelinates the useful connection).
        # If reward is NEGATIVE (further from dot), subtracting it INCREASES the delay (punishes).
        new_delays = self.connection.delay.float() - delay_update
        
        # 6. Apply constant demyelination (Catastrophic forgetting)
        new_delays += 0.00005
        
        # 7. Enforce integer timestep bounds [0, max_delay]
        self.connection.delay = torch.clamp(new_delays, 0.0, self.max_delay).long()

##########################################################################################
###### 


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

    # For each episode...
    for ep in range(episodes):
        # Reset variables for new episode.
        total_reward = 0
        rewards = np.zeros(steps)
        intercepts = 0
        step = 0
        net.reset_state_variables()
        env.reset()
        done = False
        env.render()
        clock = time.time()

        # Initialize action tensor, network output monitor, and spike train record.
        action = torch.randint(low=0, high=env.action_space.n, size=(1,))[0]
        spike_record = torch.zeros(
            (steps, int(gran / dt), env.action_space.n), device=DEVICE
        )
        # perf_ax = None

        # Run through episode.
        while not done:
            step += 1
            prev_dist = np.sqrt((env.netDot.row[0] - env.dots[0].row[0])**2 + 
                                (env.netDot.col[0] - env.dots[0].col[0])**2)

            obs, reward, done, intercept = env.step(action)
            obs = torch.Tensor(obs).to(DEVICE)
            
            curr_dist = np.sqrt((env.netDot.row[0] - env.dots[0].row[0])**2 + 
                                (env.netDot.col[0] - env.dots[0].col[0])**2)

            #calculate the gradient reward
            custom_reward = prev_dist - curr_dist

            # add multiplier for actual interceptions
            if intercept:
                custom_reward += 20.0  

            #overwrite the environment's default reward
            reward = torch.tensor(custom_reward, dtype=torch.float32).to(DEVICE)

            # Determine the action probabilities
            probabilities = torch.softmax(
                torch.sum(spike_record[step - 1 % steps], dim=0), dim=0
            )
            action = torch.multinomial(probabilities, num_samples=1).item()

            # Place the observations into the inputs.
            obs = obs.unsqueeze(0)
            inputs = {LAYER1: poisson(obs * 128, gran, dt, device=DEVICE)}
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

    # 1. Layers
    inpt = Input(n=dim * dim, shape=[1, 1, 1, dim, dim], traces=True)
    middle = AdaptiveLIFNodes(n=neurons, traces=True, thresh=-40.0, theta_plus = 0.1, tc_theta=50.0)
    out = AdaptiveLIFNodes(n=moveChoices, traces=True, thresh=-52.0, theta_plus = 0.1, tc_theta=50.0)


    network.add_layer(inpt, name="Input")
    network.add_layer(middle, name="Middle")
    network.add_layer(out, name="Output")

    # 2. Input -> Middle (Feedforward Trajectory Learning)
    delay_in_mid = torch.randint(0, 32, (inpt.n, middle.n), device=DEVICE)
    inpt_middle = MulticompartmentConnection(
        source=inpt, target=middle, 
        pipeline=[Weight(name="inpt_middle", value=torch.ones(inpt.n, middle.n))],
        device=DEVICE
    )
    inpt_middle.delay = delay_in_mid
    inpt_middle.learning_rule = ModulatedDelaySTDP(connection=inpt_middle, max_delay=32, nu=[1e-2, 1e-2], tc_e=1000.0)

    # 3. Middle -> Middle (Lateral Inhibition)
    w_recurrent = torch.ones(middle.n, middle.n, device=DEVICE) * -500.0
    w_recurrent.fill_diagonal_(0.0)
    delay_rec = torch.ones(middle.n, middle.n, device=DEVICE, dtype=torch.long)
    recurrent = MulticompartmentConnection(
        source=middle, target=middle, 
        pipeline=[Weight(name="recurrent", value=w_recurrent)],
        device=DEVICE
    )
    recurrent.delay = delay_rec

    # 4. Middle -> Output (Feedforward Action Mapping)
    delay_mid_out = torch.ones(middle.n, out.n, device=DEVICE, dtype=torch.long)
    middle_out = MulticompartmentConnection(
        source=middle, target=out, 
        pipeline=[Weight(name="middle_out", value=torch.ones(middle.n, out.n))],
        device=DEVICE
    )
    middle_out.delay = delay_mid_out
    middle_out.learning_rule = ModulatedDelaySTDP(connection=middle_out, max_delay=32, nu=[1e-2, 1e-2], tc_e=1000.0)

    network.add_connection(inpt_middle, source="Input", target="Middle")
    network.add_connection(recurrent, source="Middle", target="Middle")
    network.add_connection(middle_out, source="Middle", target="Output")
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