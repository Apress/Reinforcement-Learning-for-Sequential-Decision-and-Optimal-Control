from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from itertools import cycle
from config import PlotConfig
import numpy as np
import torch

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def smooth(data, a=0.5):
    data = np.array(data).reshape(-1, 1)
    for ind in range(data.shape[0] - 1):
        data[ind + 1, 0] = data[ind, 0] * (1-a) + data[ind + 1, 0] * a
    return data

def numpy2torch(input, size):
    """

    Parameters
    ----------
    input

    Returns
    -------

    """
    u = np.array(input, dtype='float32').reshape(size)
    return torch.from_numpy(u)

def step_relative(statemodel, state, u):
    """

    Parameters
    ----------
    state_r
    u_r

    Returns
    -------

    """
    x_ref = statemodel.reference_trajectory(state[:, -1])
    state_r = state.detach().clone()  # relative state
    state_r[:, 0:4] = state_r[:, 0:4] - x_ref
    state_next, deri_state, utility, F_y1, F_y2, alpha_1, alpha_2 = statemodel.step(state, u)
    state_r_next_bias, _, _, _, _, _, _ = statemodel.step(state_r, u) # update by relative value
    state_r_next = state_r_next_bias.detach().clone()
    state_r_next_bias[:, [0, 2]] = state_next[:, [0, 2]]            # y psi with reference update by absolute value
    x_ref_next = statemodel.reference_trajectory(state_next[:, -1])
    state_r_next[:, 0:4] = state_r_next_bias[:, 0:4] - x_ref_next
    return state_next.clone().detach(), state_r_next.clone().detach(), x_ref.detach().clone()

def recover_absolute_state(state_r_predict, x_ref, length=None):
    if length is None:
        length = state_r_predict.shape[0]
    # c = DynamicsConfig()
    ref_predict = [x_ref]
    for i in range(length-1):
        ref_t = np.copy(ref_predict[-1])
        # ref_t[0] += c.u * c.Ts * np.tan(x_ref[2])
        ref_predict.append(ref_t)
    state = state_r_predict[:, 0:4] + ref_predict
    return state, np.array(ref_predict)

def calRelError(ADP, MPC, title, simu_dir, isPlot = False):
    maxMPC = np.max(MPC, 0)
    minMPC = np.min(MPC, 0)
    relativeError = np.abs((ADP - MPC)/(maxMPC - minMPC + 1e-3))
    relativeErrorMax = np.max(relativeError, 0)
    relativeErrorMean = np.mean(relativeError, 0)
    print(title +' Error | Mean: {:.4f}%, Max: {:.4f}%'.format(relativeErrorMean*100, relativeErrorMax*100))
    if isPlot == True:
        plt.figure()
        data = relativeError
        plt.hist(data, bins=30, weights = np.zeros_like(data) + 1 / len(data))
        plt.xlabel('Relative Error of '+title)
        plt.ylabel('Frequency')
        plt.title('Relative Error of '+title)
        plt.savefig(simu_dir + '/relative-error-'+title+'.png')
        plt.close()
    return relativeErrorMean, relativeErrorMax

def idplot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title,
           figsize_scalar=1, denseMark = False, isError = False,
           legend=None,
           legend_loc="best",
           ncol=1,
           xlabel=None,
           ylabel=None,
           condition='real'
           ):
    from config import PlotConfig
    fig_size = (PlotConfig.fig_size * figsize_scalar, PlotConfig.fig_size * figsize_scalar)
    _, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=PlotConfig.dpi)
    colorList = ['darkorange', 'green', 'blue', 'red', 'yellow']
    # linestyle = 'dashed' if condition == 'virtual' else 'solid'
    linestyle = 'solid'
    linewidth = 1 if condition == 'virtual' else 1.5
    if denseMark == True:
        # markerList = ['None', 'None', 'None', 'None', 'None']
        markerList = 5 * ['.']
        markevery = 10
    else:
        markerList = ['None', 'd', '|', 'x', 'None']
        markevery = 50

    if denseMark == True:
        markevery = [j * 10 for j in range(len(xADP) // 10)] + [-1]
    plt.plot(xADP, yADP, linestyle='--',marker=markerList[0], markersize=4, zorder=10, markevery=markevery
             ,color=colorList[0], linewidth=linewidth)  # color=colorList[-1],
    for i in range(3):
        if denseMark == True:
            markevery = [j * 10 for j in range(len(xMPC[i]) // 10)] + [-1]
        plt.plot(xMPC[i], yMPC[i], linestyle = linestyle, marker=markerList[i + 1], markersize=4, markevery=markevery,
                 color=colorList[i + 1], linewidth=linewidth) # color = colorList[i],
    # if condition == 'real'and not isError:
    #     plt.plot(xMPC[i], yMPC[i], linestyle='dotted', color='grey', markevery=markevery)  # color = colorList[i],
    if condition == 'real':
        labels = ['ADP'] + ['MPC-'+str(mpcStep) for mpcStep in MPCStep]
    else:
        labels =['ADP'] + ['MPC-'+str(mpcStep) for mpcStep in MPCStep] + [ 'Ref'] if isError else ['ADP'] + ['MPC-'+str(mpcStep) for mpcStep in MPCStep]
    if condition == 'virtual':
        if 'Phase' not in title:
            plt.plot([np.min(xADP), np.max(xADP)], [0,0], linewidth = 1, color = 'grey', linestyle = '--')
        else:
            plt.scatter([0.], [0.], marker='*', s=40, c='grey')
    plt.legend(labels=labels, ncol=ncol, prop=PlotConfig.legend_font)
    plt.xlabel(xName, PlotConfig.label_font)
    plt.ylabel(yName, PlotConfig.label_font)
    plt.tick_params(labelsize=PlotConfig.tick_size)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(PlotConfig.tick_label_font) for label in labels]
    plt.savefig(simu_dir + '/' + title + '.png', bbox_inches='tight')
    plt.close()


