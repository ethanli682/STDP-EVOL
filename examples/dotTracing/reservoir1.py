import matplotlib
matplotlib.use("Agg")  # headless backend: must be set before any pyplot import,
                        # including the transitive one inside bindsnet.analysis.plotting.
                        # Required for running on SLURM / any node with no display.

import argparse
import math
import os
import time

import numpy as np
import torch
from PIL import Image

from bindsnet.analysis.plotting import plot_spikes  # plot_performance
# from bindsnet.encoding import bernoulli
from bindsnet.encoding import poisson
from bindsnet.environment.dot_simulator import DotSimulator

# from bindsnet.pipeline import EnvironmentPipeline
# from bindsnet.learning import MSTDP
from bindsnet.learning.MCC_learning import MSTDPET
from bindsnet.network import Network, network
from bindsnet.network.topology_features import Weight, Delay


from bindsnet.analysis.plotting import plot_spikes  # plot_performance
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
parser.add_argument("--neurons", type=int, default=150)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--trn_eps", type=int, default=2000)
parser.add_argument("--tst_eps", type=int, default=1000)
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
parser.add_argument("--render_replays", type=bool, default=True)
parser.add_argument("--render_every", type=int, default=10)
parser.add_argument("--replay_fps", type=int, default=20)
parser.add_argument("--view_r", type=int, default=14)

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
render_replays = args.render_replays  # save game replay GIFs
render_every = args.render_every  # save a replay every N episodes
replay_fps = args.replay_fps  # playback speed of saved replay GIFs
view_r = args.view_r  # radius of local view window around the agent

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

LAYER2 = "Middle"

LAYER3 = "Output"

# file path for recording grid observations, rewards, and performance.
OUT_FILE_PATH = "weighted_MLOLreservoir4/"




def getLocalView(obs, row, col, view_r):
    """
    Return a flattened (2*view_r+1)^2 - 1 array of the cells surrounding
    (row, col) in obs. The center cell (the agent itself) is excluded.
    Edge cases are zero-padded automatically.
    """
    padded = np.pad(obs, view_r, mode="constant", constant_values=0)
    patch = padded[row : row + 2 * view_r + 1, col : col + 2 * view_r + 1]
    flat = patch.flatten()
    center = view_r * (2 * view_r + 1) + view_r
    return np.concatenate([flat[:center], flat[center + 1 :]])

def plotWeightGraphs(weight_features, ep, out_path=OUT_FILE_PATH):
    os.makedirs(out_path, exist_ok=True)

    names = list(weight_features.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5))
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        data = weight_features[name].value.detach().cpu().numpy()
        vabs = max(abs(data.min()), abs(data.max())) or 1.0
        im = ax.imshow(data, cmap="RdBu_r", vmin=-vabs, vmax=vabs, aspect="auto")
        ax.set_title(name)
        ax.set_xlabel("target neuron")
        ax.set_ylabel("source neuron")

        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label("weight")

    fig.suptitle(f"Synaptic Weights — Episode {ep}")
    fig.tight_layout()

    fname = os.path.join(out_path, f"weights_ep{ep:05d}.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname

def plotDelayGraphs(delay_features, ep, out_path=OUT_FILE_PATH, max_delay=1):
    """
    Saves one PNG per call, containing a heatmap of every learnable delay matrix
    (one subplot per connection). Rows = source neuron, columns = target neuron,
    color = current delay in timesteps. Called every 10 episodes from runSimulator.
    """
    os.makedirs(out_path, exist_ok=True)
 
    names = list(delay_features.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5))
    if len(names) == 1:
        axes = [axes]
 
    for ax, name in zip(axes, names):
        data = delay_features[name].value.detach().cpu().numpy()
        im = ax.imshow(data, cmap="viridis", vmin=0, vmax=max_delay, aspect="auto")
        ax.set_title(name)
        ax.set_xlabel("target neuron")
        ax.set_ylabel("source neuron")
 
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label("delay (timesteps)")
 
    fig.suptitle(f"Learned Synaptic Delays \u2014 Episode {ep}")
    fig.tight_layout()
 
    fname = os.path.join(out_path, f"delays_ep{ep:05d}.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname

def captureFrame(fig=None):
    """
    Grabs the current matplotlib figure as an RGB uint8 array directly off its
    canvas -- no display required, works fine under the Agg backend. Call this
    right after env.render() so the figure being captured is the game grid and
    not some other figure (e.g. plot_spikes' own figure).
    """
    if fig is None:
        fig = plt.gcf()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return buf[:, :, :3].copy()  # drop alpha; copy since the canvas buffer is reused

def saveReplayGIF(frames, fname, fps=20):
    """
    Writes a list of RGB uint8 frames into a single animated GIF -- one file,
    scrubbable/playable as a "replay" in any image viewer or browser, no
    special tooling needed to open it (unlike e.g. a raw frame-array .npz).
    """
    if not frames:
        return None
    images = [Image.fromarray(f) for f in frames]
    images[0].save(
        fname,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    return fname

def genFileName(ftype, suffix=""):
    """
    Generates output file names for rewards and performance
    """
    # Grab system time and trim off extra large parts of the number.
    sysTime = time.time()
    sysTime = int(1e10 * (sysTime - 1e6 * (sysTime // 1e6)))

    # Create filename if one isn't provided.
    return OUT_FILE_PATH + ftype + "_s" + str(sysTime) + "_" + suffix + ".csv"


def runSimulator(
    net, env, spikes, episodes, gran=100, rfname="", pfname="",
    weight_features=None,
    render_replays=False, render_every=1, replay_fps=20, replay_prefix="replay",
    delay_log_fname="", view_r=5,
):
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

        #get current frame for replay
        # capturing = render_replays and (ep % render_every == 0)
        replay_frames = [] 
        replay_frames.append(captureFrame())

        # Initialize action tensor, network output monitor, and spike train record.
        action = torch.randint(low=0, high=env.action_space.n, size=(1,))[0]
        spike_record = torch.zeros(
            (steps, int(gran / dt), env.action_space.n), device=DEVICE
        )
        # perf_ax = None

        # Run through episode.
        while not done:
            step += 1
            ## reward: sign set by outcome (closer/farther), magnitude scaled by alignment ##
            # reward is scaled by the cosine of the angle between two vectors
            # --- positions BEFORE the step ---
            net_prev_row, net_prev_col = env.netDot.row[0], env.netDot.col[0]
            prev_dist = np.sqrt((net_prev_row - env.dots[0].row[0])**2 +
                                (net_prev_col - env.dots[0].col[0])**2)

            # step
            obs, reward, done, intercept = env.step(action)

            # erase agent trail from the network input only
            net_obs = obs.copy()
            net_obs[env.netDot.row, env.netDot.col] = 0.0
            obs = torch.Tensor(
                getLocalView(net_obs, env.netDot.row[0], env.netDot.col[0], view_r)
            ).to(DEVICE)

            # --- positions AFTER the step ---
            net_curr_row, net_curr_col = env.netDot.row[0], env.netDot.col[0]
            tar_curr_row, tar_curr_col = env.dots[0].row[0], env.dots[0].col[0]
            curr_dist = np.sqrt((net_curr_row - tar_curr_row)**2 +
                                (net_curr_col - tar_curr_col)**2)

            # signed outcome: + when the agent got closer, - when it got farther
            delta = prev_dist - curr_dist

            # alignment: cosine between the move vector and the direction to the prey's
            # CURRENT position (measured from where the agent started the step).
            dr_to_prey = tar_curr_row - net_prev_row
            dc_to_prey = tar_curr_col - net_prev_col
            dr_moved   = net_curr_row - net_prev_row
            dc_moved   = net_curr_col - net_prev_col

            dir_mag  = np.sqrt(dr_to_prey**2 + dc_to_prey**2)
            move_mag = np.sqrt(dr_moved**2 + dc_moved**2)

            if move_mag < 1e-8 or dir_mag < 1e-8:
                # agent didn't move (or already on the prey): alignment undefined -> neutral
                alignment = 0.0
            else:
                alignment = (dr_moved * dr_to_prey + dc_moved * dc_to_prey) / (dir_mag * move_mag)
            alignment = float(np.clip(alignment, -1.0, 1.0))   # guaranteed cosine in [-1, 1]

            # scale in [0, 1]: 1.0 = perfectly aimed, 0.0 = aimed opposite
            scale = 0.5 + 0.5 * alignment

            if delta >= 0:
                # got closer: reward grows with good aim  (aligned -> big, poor aim -> medium)
                custom_reward = delta * (0.5 + scale)      # multiplier in [0.5, 1.5]
            else:
                # got farther: penalty grows with bad aim, stays negative regardless of aim
                custom_reward = delta * (1.5 - scale)      # multiplier in [0.5, 1.5], delta<0

            # small standalone directional term so good aim still earns signal when the
            # agent barely moved (delta near 0). Set weight to 0.0 to disable.
            custom_reward += 0.2 * alignment

            # large flat bonus for an actual interception, independent of geometry
            if intercept:
                custom_reward += 20.0

            reward = torch.tensor(custom_reward, dtype=torch.float32).to(DEVICE)

            
            # get output neuron proabilities and get an action
            probabilities = torch.softmax(
                torch.sum(spike_record[(step - 1) % steps], dim=0), dim=0
            )
            action = torch.multinomial(probabilities, num_samples=1).item()

            # Place the observations into the input layer
            obs = obs.unsqueeze(0)
            inputs = {LAYER1: poisson(obs * 150, gran, dt, device=DEVICE)}
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
                if write:
                    if rfname != "":
                        with open(rfname, "ab") as f:
                            np.savetxt(f, rewards, delimiter=",", fmt="%.6f")

            spikes_ = {layer: spikes[layer].get("s").view(gran, -1) for layer in spikes}
            spike_ims, spike_axes = plot_spikes(spikes_, ims=spike_ims, axes=spike_axes)
            # perf_ax = plot_performance(reward, x_scale=10, ax=perf_ax)

            env.render()
            replay_frames.append(captureFrame())
            total_reward += reward

            if step % 10 == 0:
                print(
                    f"Iteration: {step} (Time: {time.time() - clock:.4f}); reward: {reward}"
                )
                clock = time.time()

        print(f"Episode {ep} total reward:{total_reward}")
        os.makedirs(OUT_FILE_PATH, exist_ok=True)
        replay_fname = os.path.join(OUT_FILE_PATH, f"{replay_prefix}_ep{ep:05d}.gif")
        saveReplayGIF(replay_frames, replay_fname, fps=replay_fps)
        print(f"  [replay] saved {replay_fname} ({len(replay_frames)} frames)")

        if weight_features is not None:
            saved_to = plotWeightGraphs(weight_features, ep)
            print(f"  [weight plot] saved {saved_to}")
        # Save intcercepts thus far to file
        if write:
            if rfname != "":
                with open(rfname, "a+") as f:
                    if 0 < ep:
                        f.write("," + str(rewards))
                    else:
                        f.write(str(rewards))
        if pfname != "":
            with open(pfname, "a+") as f:
                if 0 < ep:
                    f.write("," + str(intercepts))
                else:
                    f.write(str(intercepts))

        # Cycle output files every 10000 iterations
        if ep % fcycle == 0:
            env.cycleOutFiles()
 

## NOTE: only last layer of delays learns. 
def main():
    # Build network.
    network = Network(dt=dt)

    inpt_n = (2 * view_r + 1) ** 2 - 1  # surrounding cells only, center excluded
    inpt = Input(n=inpt_n,
                 shape=[1,1,1,1,inpt_n],
                 traces=True,
                 )
    
    middle = AdaptiveLIFNodes(n=neurons,
                      traces=True,
                      thresh=-52.0,
                      theta_plus=0.01,
                      tc_decay=50.0,
                      )

    # out = AdaptiveLIFNodes(n=moveChoices, traces=True, thresh=-52.0, theta_plus = 0.1, tc_theta=50.0)

    out = LIFNodes(n=moveChoices, 
                   traces=True,
                   thresh=-49.0, 
                   tc_trace=50.0,
                   tc_decay=50.0,
                   )

    network.add_layer(inpt, name="Input")
    network.add_layer(middle, name="Middle")
    network.add_layer(out, name="Output")

    feat_weight_in_mid = Weight(
        name="weight_in_mid",
        value=torch.randn(inpt.n, middle.n, device=DEVICE),
    )
    inpt_middle = MulticompartmentConnection(
        source=inpt, target=middle,
        pipeline=[feat_weight_in_mid],  # Delay MUST come first
        device=DEVICE,
    )

    # 3. Middle -> Middle  reservoir layer: 50% excitatory, 50% inhibitory
    exc_mask = torch.bernoulli(torch.full((middle.n, middle.n), 0.45, device=DEVICE))
    signs = exc_mask * 2 - 1  # +1 excitatory, -1 inhibitory
    w_recurrent = torch.rand(middle.n, middle.n, device=DEVICE) * signs * 2.0
    w_recurrent.fill_diagonal_(0.0)  # no self-connections
    recurrent = Weight(
        name="recurrent",
        value=w_recurrent,
        learning_rule=MSTDPET,
        nu=[1e-2, 1e-2]
    )

    middle_middle = MulticompartmentConnection(
        source=middle,
        target=middle,
        pipeline=[recurrent],
        device=DEVICE
    )

    feat_weight_mid_out = Weight(
        name="weight_mid_out",
        value=torch.rand(middle.n, out.n, device=DEVICE) * 4.8, 
        learning_rule=MSTDPET,
        nu=[1e-2, 1e-2],
        
    )
    middle_out = MulticompartmentConnection(
        source=middle, target=out,
        pipeline=[feat_weight_mid_out],  
        device=DEVICE,
    )

    network.add_connection(inpt_middle, source="Input", target="Middle")
    network.add_connection(middle_middle, source="Middle", target="Middle")
    network.add_connection(middle_out, source="Middle", target="Output")
    network.to(DEVICE)

    weight_features = {"inpt_middle": feat_weight_in_mid, "recurrent": recurrent, "middle_out": feat_weight_mid_out}

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
    delayFile = genFileName("delays", "train")
    environment.addFileSuffix("train")
    runSimulator(
        network,
        environment,
        spikes,
        episodes=trn_eps,
        gran=granularity,
        rfname=rewFile,
        pfname=perfFile,
        weight_features=weight_features,
        render_replays=render_replays,
        render_every=render_every,
        replay_fps=replay_fps,
        replay_prefix="replay_train",
        delay_log_fname=delayFile,
        view_r=view_r,
    )

    # Freeze learning
    network.learning = False

    print("Testing: ")
    rewFile = genFileName("rew", "test")
    perfFile = genFileName("perf", "test")
    delayFile = genFileName("delays", "test")
    environment.changeFileSuffix("train", "test")
    runSimulator(
        network,
        environment,
        spikes,
        episodes=tst_eps,
        gran=granularity,
        rfname=rewFile,
        pfname=perfFile,  
        weight_features=weight_features,
        render_replays=render_replays,
        render_every=render_every,
        replay_fps=replay_fps,
        replay_prefix="replay_test",
        delay_log_fname=delayFile,
        view_r=view_r,
    )


if __name__ == "__main__":
    main()