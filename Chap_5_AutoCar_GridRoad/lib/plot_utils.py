"""
Copyright (c). All Rights Reserved.
<Reinforcement Learning for Sequential Decision and Optimal Control> (Year 2023)
Intelligent Driving Lab (iDLab), Tsinghua University

by Shengbo Eben Li & Wenxuan Wang & Zhiqian Lan

Description: Chapter 5:  Utils for plot
"""
import matplotlib.pyplot as plt
import numpy as np
from lib.utils import nextstate, headlocation, cm2inch, default_cfg

# Boundary
laneBoundaryX = np.array([0,1,2,3,3,4,5,5,6,6,7,7,8,8,9,10,10,11,12,13,13,14,15,15,16,16,17,17,18,18,19,20,20,21,22,23,24])
laneBoundaryYH = np.array([12,12,12,12,11,11,11,10,10,9,9,8,8,7,7,7,6,6,6,6,7,7,7,8,8,9,9,10,10,11,11,11,12,12,12,12,12])
laneBoundaryYL = np.array([6,6,6,6,5,5,5,4,4,3,3,2,2,1,1,1,0,0,0,0,1,1,1,2,2,3,3,4,4,5,5,5,6,6,6,6,6])

# simulation settings
carDirection = 5
action = 3
roadColumn = 24
roadRow = 12

# Figure settings
figureS = (14,7) # figure size
figureEVN = (8,4)
fontS = 12       # fontsize 
bias = 0.2
mid = 0.45
alength = 0.5    # arrow length
awidth = 0.1     # arrow width for drawing route
alength2 = 0.9
format = 'png'

def Draw_demo():
    # just for test
    # fig, ax = plt.subplots(figsize=(12,6))
    fig = plt.figure(figsize=figureS)
    ax = fig.add_subplot(1,1,1)
    ax.plot(laneBoundaryX,laneBoundaryYL,'k-')
    ax.plot(laneBoundaryX,laneBoundaryYH,'k-')
    ax.grid(linestyle='--')
    ax.axis('scaled')
    ax.set_xticks(np.arange(0, 25, step=1))
    ax.set_yticks(np.arange(0, 13, step=1))
    ax.arrow(x=5.2,y=5.2,dx=0.5,dy=0.5,width=0.04,color='blue')
    ax.set_xlim((0,24))
    ax.set_ylim((0,12))
    # plt.show()
    return fig, ax

def Backbone():
    # to generate the map. return figure and its axes for further drawing
    fig = plt.figure(figsize=figureS)
    ax = fig.add_subplot(1,1,1,position=[0.04,0.05,0.95,0.92])
    # print(ax.get_position())
    ax.plot(laneBoundaryX,laneBoundaryYL,'k-',linewidth=4)
    ax.plot(laneBoundaryX,laneBoundaryYH,'k-',linewidth=4)
    ax.grid(linestyle='--')
    ax.axis('scaled')
    ax.set_xticks(np.arange(0, 25, step=1))
    ax.set_yticks(np.arange(0, 13, step=1))
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    ax.set_xlim((0,24))
    ax.set_ylim((0,12))
    
    return fig

def Draw_sth():
    fig = Backbone()
    axes = fig.get_axes()
    ax = axes[0]
    ax.arrow(x=5.2,y=5.2,dx=0.5,dy=0.5,width=0.04,color='blue')
    ax.text(x=6.2,y=6.2,s='1.4',fontsize=fontS)
    return fig


#-------------------------------------------
# Draw a specific route with initial states
#-------------------------------------------
def Draw_Route(policy, filepath):

    # n = 3
    s = []
    s.append(np.array([0,8,1]))
    s.append(np.array([7,2,3]))
    s.append(np.array([17,8,0]))
    # s.append(np.array([2,7,0]))
    # s.append(np.array([8,2,4]))
    # s.append(np.array([15,6,0]))
    fig = Backbone()
    # str_name = 'Route'
    axes = fig.get_axes()
    ax = axes[0]
    for s_ in s:
        s_i = s_.copy()
        s_h = headlocation(s_i)
        # q1 = s_h[0] - s_i[0]
        # q2 = s_h[1] - s_i[1]
        ax.plot(s_i[0]+0.5,s_i[1]+0.5,marker='s', 
                                    markerfacecolor='xkcd:light green', 
                                    markeredgecolor='black', 
                                    markeredgewidth=4, 
                                    markersize=34)
        ax.plot(s_h[0]+0.5,s_h[1]+0.5,marker='s', 
                                    markerfacecolor='yellow', 
                                    markeredgecolor='black', 
                                    markeredgewidth=4, 
                                    markersize=34)
        ax.plot([s_i[0]+0.5,s_h[0]+0.5],[s_i[1]+0.5,s_h[1]+0.5],linewidth=3,color='k')
        flag1 = 0
        flag2 = 0
        for _ in range(1000):
            s_h = headlocation(s_i)
            if policy[s_i[0],s_i[1],s_i[2]] == -1:
                flag1 = 1
            if s_h[0] == roadColumn - 1 or s_i[0] == roadColumn - 1:
                flag2 = 1
            action = policy[s_i[0],s_i[1],s_i[2]]
            s_next = nextstate(s_i,action)
            if action == 0:
                color = 'red'
            elif action == 1:
                color = 'blue'
            else:
                color = 'xkcd:cyan'
            if s_next[2] == 0:
                ax.arrow(x=s_i[0]+0.5,y=s_i[1]+0.5,dx=0,dy=alength2,width=0.04,color=color)
            if s_next[2] == 1:
                ax.arrow(x=s_i[0]+0.5,y=s_i[1]+0.5,dx=alength2,dy=alength2,width=0.04,color=color)
            if s_next[2] == 2:
                ax.arrow(x=s_i[0]+0.5,y=s_i[1]+0.5,dx=alength2,dy=0,width=0.04,color=color)
            if s_next[2] == 3:
                ax.arrow(x=s_i[0]+0.5,y=s_i[1]+0.5,dx=alength2,dy=-alength2,width=0.04,color=color)
            elif s_next[2] == 4:
                ax.arrow(x=s_i[0]+0.5,y=s_i[1]+0.5,dx=0,dy=-alength2,width=0.04,color=color)
            if flag1 == 1 or flag2 == 1:
                break
            # ss = s_i.copy()
            s_i = s_next
        ax.plot(s_i[0]+0.5,s_i[1]+0.5,marker='s', 
                                    markerfacecolor='xkcd:light green', 
                                    markeredgecolor='black', 
                                    markeredgewidth=4, 
                                    markersize=34)
        ax.plot(s_next[0]+0.5,s_next[1]+0.5,marker='s', 
                                    markerfacecolor='yellow', 
                                    markeredgecolor='black', 
                                    markeredgewidth=4, 
                                    markersize=34)

    fig.savefig(('%s/Route.' % filepath) + format)
    return

#-------------------------------------------
# Draw value
#-------------------------------------------
def Draw_Value(V_Q, policy, Using_V_Value, filepath):
    # draw value in given figure
    for s3 in range(carDirection):
        fig = Backbone()
        axes = fig.get_axes()
        ax = axes[0]
        for s1 in range(roadColumn):
            for s2 in range(roadRow):
                s = np.array([s1,s2,s3])
                if policy[tuple(s)] != -1:
                    if Using_V_Value == 1:
                        valueText = str(round(V_Q[tuple(s)]*10)/10)
                    else:
                        valueText = str(round(V_Q[s1,s2,s3,policy[s1,s2,s3]]*10)/10)
                    ax.text(x=s1+0.2,y=s2+0.35,s=valueText,fontsize=fontS)

        fig.savefig(('%s/Value_%i.'%(filepath, s3+1))+format)
    return


#-------------------------------------------
# Draw optimal policy
#-------------------------------------------
def Draw_Policy(policy, filepath):
    # draw optimal policy in given figure

    for s3 in range(carDirection):
        fig = Backbone()
        axes = fig.get_axes()
        ax = axes[0]
        for s1 in range(roadColumn):
            for s2 in range(roadRow):
                s = np.array([s1,s2,s3])

                if policy[tuple(s)] != -1:
                    s_next = nextstate(s,policy[tuple(s)])
                else:
                    s_next = np.array([-1,-1,-1])
                
                if s_next[2] == 0:
                    ax.arrow(x=s1+mid,y=s2+bias,dx=0,dy=alength,width=0.04,color='blue')
                elif s_next[2] == 1:
                    ax.arrow(x=s1+bias,y=s2+bias,dx=alength,dy=alength,width=0.04,color='blue')
                elif s_next[2] == 2:
                    ax.arrow(x=s1+bias,y=s2+mid,dx=alength,dy=0,width=0.04,color='blue')
                elif s_next[2] == 3:
                    ax.arrow(x=s1+bias,y=s2+1-bias,dx=alength,dy=-alength,width=0.04,color='blue')
                elif s_next[2] == 4:
                    ax.arrow(x=s1+mid,y=s2+1-bias,dx=0,dy=-alength,width=0.04,color='blue')
        fig.savefig(('%s/Policy_%i.'%(filepath, s3+1))+format)
    return

def EVN_DP_step(iteration, step_size, DP_reward_n, std, savepath, mode=True):
    fig = plt.figure(figsize=cm2inch(default_cfg.fig_size),dpi=default_cfg.dpi)
    ax = fig.add_subplot(1,1,1)
    ax.set_xlim([0,iteration.size-1])
    # ax.set_ylim([-23,-14])
    ax.set_ylabel('Total Average Return',fontdict=default_cfg.label_font)
    ax.set_xlabel('Cycles',fontdict=default_cfg.label_font)
    for i in range(len(step_size)):
        ax.plot(iteration,DP_reward_n[i])
        ax.fill_between(iteration, DP_reward_n[i] - std[i], DP_reward_n[i] + std[i], alpha=0.5, antialiased=True)
    if mode:
        ax.legend([r'Step/PEV='+str(s) for s in step_size],loc='lower right', ncol=1, prop=default_cfg.legend_font)
    ax.plot(iteration, -14.6*np.ones_like(iteration), 'r--')
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(default_cfg.tick_label_font) for label in labels]
    plt.tight_layout()
    if mode:
        fig.savefig(('%s/DP_step_size_Rewards.'%savepath)+format)
    else:
        fig.savefig(('%s/DP_Rewards.'%savepath)+format)

def EVN_DP_gamma(iteration, gamma_size, DP_reward_gamma, std, savepath):
    fig = plt.figure(figsize=cm2inch(default_cfg.fig_size),dpi=default_cfg.dpi)
    ax = fig.add_subplot(1,1,1)
    ax.set_xlim([0,iteration.size-1])
    # ax.set_ylim([-20,-14])
    ax.set_ylabel('Total Average Return',fontdict=default_cfg.label_font)
    ax.set_xlabel('Cycles',fontdict=default_cfg.label_font)
    for i in range(len(gamma_size)):
        ax.plot(iteration,DP_reward_gamma[i])
        ax.fill_between(iteration, DP_reward_gamma[i] - std[i], DP_reward_gamma[i] + std[i], alpha=0.5, antialiased=True)
    ax.legend([r'$\gamma$='+str(s) for s in gamma_size],loc='lower right', ncol=1, prop=default_cfg.legend_font)
    ax.plot(iteration, -14.6 * np.ones_like(iteration), 'r--')
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(default_cfg.tick_label_font) for label in labels]
    plt.tight_layout()
    fig.savefig(('%s/DP_gamma_Rewards.'%savepath)+format)

def EVN_DP_RMSE(name, iteration, Size, DP_RMSE, std, savepath, mode=True):
    fig = plt.figure(figsize=cm2inch(default_cfg.fig_size),dpi=default_cfg.dpi)
    ax = fig.add_subplot(1,1,1)
    ax.set_xlim([0,iteration.size-1])
    # ax.set_ylim([0,12])
    ax.set_ylabel('RMS Error',fontdict=default_cfg.label_font)
    ax.set_xlabel('Cycles',fontdict=default_cfg.label_font)
    if name == 'DP_step_size_RMSE':
        legend_str = r'Step/PEV='
    else:
        legend_str = r'$\gamma$='

    for i in range(len(Size)):
        ax.plot(iteration,DP_RMSE[i])
        ax.fill_between(iteration, DP_RMSE[i] - std[i], DP_RMSE[i] + std[i], alpha=0.5, antialiased=True)
    if mode:
        ax.legend([legend_str+str(s) for s in Size],loc='best', ncol=1, prop=default_cfg.legend_font)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(default_cfg.tick_label_font) for label in labels]
    plt.tight_layout()
    fig.savefig(('%s/%s.'%(savepath,name))+format)

def EVN_TD(X, QL=None, Sarsa=None, stdQL=None, stdSA=None, savepath=None, name=None, mode=True):
    fig = plt.figure(figsize=cm2inch(default_cfg.fig_size),dpi=default_cfg.dpi)
    ax = fig.add_subplot(1,1,1)
    X = X / 1000
    ax.set_xlim([0,X[-1]])
    ax.set_xlabel(r'Episodes [$\times 10^3$]', fontdict=default_cfg.label_font)
    ax.plot(X, QL)
    if mode:
        ax.plot(X, Sarsa)
        ax.fill_between(X, QL-stdQL, QL+stdQL, alpha=0.5, antialiased=True)
        ax.fill_between(X, Sarsa-stdSA, Sarsa+stdSA, alpha=0.5, antialiased=True)
    if mode:
        if name == 'QL-SARSA_Rewards':
            ax.legend(['Q-learn','Sarsa'], loc='lower right', ncol=1, prop=default_cfg.legend_font)
            # ax.set_ylim([-23,-15])
            ax.set_ylabel('Total Average Reward',fontdict=default_cfg.label_font)
            ax.plot(X, -14.6 * np.ones_like(X), 'r--')
        else:
            ax.legend(['Q-learn','Sarsa'],loc='best', ncol=1, prop=default_cfg.legend_font)
            # ax.set_ylim([0,20])
            ax.set_ylabel('RMS Error',fontdict=default_cfg.label_font)
    else:
        if name == 'Rewards':
            ax.set_ylabel('Total Average Reward',fontdict=default_cfg.label_font)
            ax.plot(X, -14.6 * np.ones_like(X), 'r--')
        else:
            ax.set_ylabel('RMS Error',fontdict=default_cfg.label_font)

    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(default_cfg.tick_label_font) for label in labels]
    plt.tight_layout()
    fig.savefig(('%s/%s.'%(savepath,name))+format)

def EVN_Sarsa_step(X, Sarsa_data, Sarsa_std, Size, savepath:str, name:str):
    fig = plt.figure(figsize=cm2inch(default_cfg.fig_size),dpi=default_cfg.dpi)
    ax = fig.add_subplot(1,1,1)
    X = X / 1000
    ax.set_xlim([0,X[-1]])
    ax.set_xlabel(r'Episodes [$\times 10^3$]', fontdict=default_cfg.label_font)
    for i in range(len(Size)):
        ax.plot(X, Sarsa_data[i])
        ax.fill_between(X, Sarsa_data[i]-Sarsa_std[i], Sarsa_data[i]+Sarsa_std[i], alpha=0.5, antialiased=True)
    if name == 'Sarsa_Rewards':
        name = 'Sarsa_n_step_Rewards'
        ax.set_ylabel('Total Average Reward',fontdict=default_cfg.label_font)
        ax.plot(X, -14.6 * np.ones_like(X), 'r--')
        ax.legend([r'Pair/PEV=' + str(s) for s in Size], loc='lower right', ncol=1, prop=default_cfg.legend_font)
    else:
        name = 'Sarsa_n_step_RMSE'
        ax.set_ylabel('RMS Error',fontdict=default_cfg.label_font)
        ax.legend([r'Pair/PEV='+str(s) for s in Size],loc='best', ncol=1, prop=default_cfg.legend_font)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(default_cfg.tick_label_font) for label in labels]
    plt.tight_layout()
    fig.savefig(('%s/%s.'%(savepath,name))+format)

def EVN_QL_Alpha(X, QL_data, QL_std, Size, savepath:str, name:str):
    fig = plt.figure(figsize=cm2inch(default_cfg.fig_size),dpi=default_cfg.dpi)
    ax = fig.add_subplot(1,1,1)
    X = X / 1000
    ax.set_xlim([0,X[-1]])
    ax.set_xlabel(r'Episodes [$\times 10^3$]', fontdict=default_cfg.label_font)
    for i in range(len(Size)):
        ax.plot(X, QL_data[i])
        ax.fill_between(X, QL_data[i]-QL_std[i], QL_data[i]+QL_std[i], alpha=0.5, antialiased=True)
    if name == 'QL_Rewards':
        name = 'QL_alpha_Rewards'
        ax.set_ylabel('Total Average Reward',fontdict=default_cfg.label_font)
        ax.plot(X, -14.6 * np.ones_like(X), 'r--')
        ax.legend([r'$\alpha$=' + str(s) for s in Size], loc='lower right', ncol=1, prop=default_cfg.legend_font)
    else:
        name = 'QL_alpha_RMSE'
        ax.set_ylabel('RMS Error',fontdict=default_cfg.label_font)
        ax.legend([r'$\alpha$='+str(s) for s in Size],loc='best', ncol=1, prop=default_cfg.legend_font)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(default_cfg.tick_label_font) for label in labels]
    plt.tight_layout()
    fig.savefig(('%s/%s.'%(savepath,name))+format)

if __name__ == '__main__':
    print('This is the plot lib for RL example "Autocar"...')
    fig = Draw_sth()
    plt.show()
    #fig.savefig('demo2.jpg')