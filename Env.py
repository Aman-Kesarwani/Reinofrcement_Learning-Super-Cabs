# Import routines

import numpy as np
import math
import random
from itertools import permutations

# Defining hyperparameters
m = 5  # number of cities, ranges from 0 ..... m-1
t = 24  # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5  # Per hour fuel and other costs
R = 9  # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = [(0, 0)] + \
            list(permutations([item for item in range(m)], 2))
        self.state_space = [[x, y, z]
                            for x in range(m) for y in range(t) for z in range(d)]
        self.state_init = random.choice(self.state_space)
        # Start the first round
        self.reset()

    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. 
        This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""

        state_vec = [0 for _ in range(m+t+d)]
        state_vec[self.state_get_loc(state)] = 1
        state_vec[m+self.state_get_time(state)] = 1
        state_vec[m+t+self.state_get_day(state)] = 1

        return state_vec

    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        if location == 1:
            requests = np.random.poisson(12)
        if location == 2:
            requests = np.random.poisson(4)
        if location == 3:
            requests = np.random.poisson(7)
        if location == 4:
            requests = np.random.poisson(8)

        sitting_idle = [0]
		
        if requests > 15:
            requests = 15
			
        # the driver can sit idle and is free to refuse all customer requests. Add the index of action (0,0).
        possible_actions_index = random.sample(range(1, (m-1)*m + 1), requests) + sitting_idle
        actions = [self.action_space[i] for i in possible_actions_index]

        return possible_actions_index, actions

    def recondition_time_day(self, time, day, riding_duration):
        
        #Returns the time and day post the driver's journey. 
    
        riding_duration = int(riding_duration)

        if (time + riding_duration) < 24:
            time = time + riding_duration
        else:
            # lets convert the time to 0-23 range
            time = (time + riding_duration) % 24 
            
            # Get the number of days
            num_days = (time + riding_duration) // 24
            
            # Convert the day to 0-6 range
            day = (day + num_days ) % 7

        return time, day
    
    def next_state_func(self, state, action, Time_matrix):
        #Takes state and action as input and returns next state"""
        next_state = []
        
        total_time = 0

        # time to go from current  location to pickup location
        curr_loc_to_pickup_loc_time = 0  

        # idle time - driver chooses to refuse all requests
        waiting_time = 0 

        # time taken from Pick-up to drop 
        riding_time = 0    
        
        # get the current location, time, day and request locations
        curr_location = self.state_get_loc(state)
        pickup_location = self.action_get_pickup(action)
        drop_location = self.action_get_drop(action)
        current_time = self.state_get_time(state)
        current_day = self.state_get_day(state)
           
        # Lets check various cases: 
        # 1) Driver refuse all requests 2) Driver is at the pickup point 3) Driver is not at pick up point

        if ((pickup_location== 0) and (drop_location == 0)):
            # wait time is 1 unit if driver refuse all requests, next location is current location
            waiting_time = 1
            next_loc = curr_location

        elif (curr_location == pickup_location):
            # driver is already at pickup point, waiting time and time to reach to pickup are both 0
            riding_time = Time_matrix[curr_location][drop_location][current_time][current_day]
            
            # next location is the drop location
            next_loc = drop_location
        
        else:
            # Driver not at the pickup point, he would travel to pickup point first
            # time take to reach pickup point
            curr_loc_to_pickup_loc_time      = Time_matrix[curr_location][pickup_location][current_time][current_day]
            new_time, new_day = self.recondition_time_day(current_time, current_day, curr_loc_to_pickup_loc_time)
            
            # The driver is now at the pickup point, Lets calculate the time taken to drop the passenger
            riding_time = Time_matrix[pickup_location][drop_location][new_time][new_day]
            next_loc  = drop_location

        # Lets evaluate the total time
        total_time = (waiting_time + curr_loc_to_pickup_loc_time + riding_time)
        next_time, next_day = self.recondition_time_day(current_time, current_day, total_time)
        
        # derive the next_state using the next_loc and the new time states.
        next_state = [next_loc, next_time, next_day]
        
        return next_state, waiting_time, curr_loc_to_pickup_loc_time, riding_time
    

    def reset(self):
        return self.action_space, self.state_space, self.state_init

    def reward_func(self, waiting_time, curr_loc_to_pickup_loc_time, riding_time):
        #Takes in state, action and Time-matrix and returns the reward
        
        passenger_time = riding_time
        idle_time      = waiting_time + curr_loc_to_pickup_loc_time
        
        reward = (R * passenger_time) - (C * (passenger_time + idle_time))

        return reward

    def step(self, state, action, Time_matrix):    
        # Get the next state and the other time durations
        next_state, waiting_time, curr_loc_to_pickup_loc_time, riding_time = self.next_state_func(
            state, action, Time_matrix)

        # Lets Evaluate the reward based on the different time durations
        rewards = self.reward_func(waiting_time, curr_loc_to_pickup_loc_time, riding_time)
        total_time = waiting_time + curr_loc_to_pickup_loc_time + riding_time
        
        return rewards, next_state, total_time


    #getter ans setters for members

    def state_get_loc(self, state):
        return state[0]

    def state_set_loc(self, state, loc):
        state[0] = loc

    def state_get_time(self, state):
        return state[1]

    def state_set_time(self, state, time):
        state[1] = time

    def state_get_day(self, state):
        return state[2]

    def state_set_day(self, state, day):
        state[2] = day

    def action_get_pickup(self, action):
        return action[0]

    def action_set_pickup(self, action, pickup):
        action[0] = pickup

    def action_get_drop(self, action):
        return action[1]

    def action_set_drop(self, action, drop):
        action[1] = drop

 
    

   

  

    
