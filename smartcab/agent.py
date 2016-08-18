import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.qtable = {}
        self.diminish_alpha = False
        self.diminish_epsilon = False

        self.alpha = 0.75
        self.alpha_factor = 0.98
        self.epsilon = 1.0
        self.epsilon_factor = 0.975
        self.gamma = 0.0
        self.total_reward = 0
        self.last_state = None
        self.state = None
        self.last_reward = 0
        self.last_action = None
        self.trial = 1

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        if (self.diminish_epsilon):
            self.epsilon *= self.epsilon_factor
        elif (self.trial  >= 90):
            self.epsilon = 0

        if (self.diminish_alpha):
            self.alpha *= self.alpha_factor

        self.total_reward = 0
        self.last_action = None
        self.last_state = None
        self.state = None
        self.last_reward = 0
        self.trial += 1


    # new_experiment is only used during grid search
    def new_experiment(self, alpha, gamma):
        self.reset()
        self.alpha = alpha

        if self.diminish_epsilon:
            self.epsilon = epsilon
        else:
            self.epsilon = 1.0


        self.gamma = gamma
        self.trial = 0


    def get_max_q(self, a_state):
        max_q = 0
        subtable = self.qtable[a_state]
        for a in (None, 'forward', 'left', 'right'):
            q_value = self.get_q(a_state, a)
            if  q_value > max_q:
                max_q = q_value

        return max_q

    def get_best_action(self, a_state):
        best_action = None
        max_q = 0
        subtable = self.qtable[a_state]
        for a in (None, 'forward', 'left', 'right'):
            q_value = self.get_q(a_state, a)
            if  q_value > max_q:
                best_action = a
                max_q = q_value

        return best_action

    def get_q(self, a_state, a_action):
        if self.qtable.has_key(a_state):
            subtable = self.qtable[a_state]
            if subtable.has_key(a_action):
                return subtable[a_action]

        return 0


    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = (inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint)

        # TODO: Select action according to your policy
        valid_actions = [None, 'forward', 'left', 'right']
        q_value = 0

        if random.random() > self.epsilon and self.qtable.has_key(self.state):
            action = self.get_best_action(self.state)
        else:
            action = random.choice(['forward', 'left', 'right']);
            if (not self.qtable.has_key(self.state)):
                self.qtable[self.state] = {'forward': 0.0, 'left': 0.0, 'right': 0.0, None: 0.0}

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.total_reward += reward

        if self.gamma == 0.0:
            q_value = self.qtable[self.state][action]
            q_value += self.alpha * (reward - q_value)
            self.qtable[self.state][action] = q_value

        elif self.last_state != None:
            max_q = self.get_max_q(self.state)
            q_value = self.qtable[self.last_state][self.last_action]
            q_value += self.alpha * (self.last_reward + self.gamma * max_q - q_value)
            self.qtable[self.last_state][self.last_action] = q_value

        self.last_action = action
        self.last_reward = reward
        self.last_state = self.state

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track

    sim = Simulator(e, update_delay=0.01, display=True)  # create simulator (uses pygame when display=True, if available)
    a.diminish_epsilon = False
    a.diminish_alpha = False
    a.alpha = 0.250
    a.gamma = 0.125
    sim.run(n_trials=100)  # run for a specified number of trials

def run_experiment(simulator, agent, alpha, gamma):
    """ Only used during grid search """
    """ simulator.py has been modified to collect statistics"""

    df = pd.DataFrame(columns=['success', 'penalty_per_step', 'efficiency']);
    print "Running experiment for alpha = {}, gamma = {}".format(alpha, gamma)

    for i in range(0,40):
        agent.new_experiment(alpha, gamma)
        simulator.run(n_trials=100)  # run for a specified number of trials
        result = simulator.calc_stats(10)
        print result
        df.loc[i] = result

    df = df.mean()
    result = pd.DataFrame(columns=['alpha', 'gamma', 'success', 'penalty_per_step', 'efficiency']);
    result.loc[0] = [alpha, gamma, df['success'], df['penalty_per_step'], df['efficiency']]
    print result
    print
    print

    return result

def grid_search():
    '''Grid serch over alpha and gamma.'''
    '''Needs modified simulator.py for collecting statistics'''

    e = Environment()
    a = e.create_agent(LearningAgent)
    e.set_primary_agent(a, enforce_deadline=True)

    sim = Simulator(e, update_delay=0.0, display=False)
    df = pd.DataFrame(columns=['alpha', 'epsilon', 'gamma', 'success', 'penalty_per_step', 'efficiency', 'deadline']);

    for alpha in (0.75, 0.5, 0.25, 0.125):
        for gamma in (0.75, 0.5, 0.25, 0.125, 0.0):
            result = run_experiment(sim, a, alpha, gamma)
            df = df.append(result, ignore_index=True)

    print df.to_csv()


if __name__ == '__main__':
    #grid_search()
    run()
