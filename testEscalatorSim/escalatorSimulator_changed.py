#!/use/bin/env python3
# This is want to simulate escalator.
# I have question: Does it get a higher throughput when people trying to walk on running escalators?
# 
# - People get lower latency when walking on escalator
# - Walking needs at least 2 spaces between persons, while standing still just 1 or 0 steps (0 means each step stands one person
#
# To formulate this process, we simplify the model:
# - From time 0, we started to generate stairs(with increment index)
# - The escalator has SPEED, TOTAL_STAIRS as parameters
# - For given time T, we can calculate exposed index interval [start_T_i, end_T_i]
# - Walking speed is 80 steps/min
# - Walking is only available when current stair in in [start_T_i, end_T_i], and has limitation of at least 2 spaces.
# - For exiting point, no limitation, means multiple person can stand in same stair.
# - Entry point always has people. Only one person allowed for one stairs at one time.
# 
# For each person, it has these attributes:
# - Time stepping on stair
# - Stair index
# - Walking or not(true means if available spaces ahead, then walking)
# - Last walking time
# - Time stepping out stair
# - Final stair index
#
# We use microsecond as simulation level.
# Copyright (C) 2018 Daochen Shi
# Author: Daochen Shi

import random


WalkingInterval = 60000/80 # 80 steps per minute(60000 ms)


class Person:
  """Simulate one person on stair"""
  def __init__(self, time_in, stair_idx, walking):
    self.time_in = time_in
    self.stair_idx = stair_idx
    self.walking = walking
    self.last_walking = time_in # last walking time is the time entered escalator
    self.time_out = time
    self.stair_final = stair_idx # the final stair, could be changed if walking

  # walking has limitations:
  # can_walk means you have enough space ahead, you can walk
  # self.walking means you rested enough and you can walk now.
  # last_stair means the last stair you can stand on,
  def simulation(self, time, can_walk, last_stair):
    """This is do simulation based on given time."""
    if can_walk and self.walking and time > self.last_walking + WalkingInterval:
      self.last_walking = time # modify last walking time
      self.stair_final -= 1 # one step forward
      return True

    # if you step out of range, you stop
    if self.stair_final < last_stair :
      self.walking = False
      self.time_out = time
#      print "you have to out stair", self
      return False
    # still in stair, can do next simulation
    return True

  # Get string of this object
  def __str__(self):
    return "time_in={}, stair_idx={}, walking={}, last_walking={}, time_out={}, stair_final={}".format(self.time_in, self.stair_idx, self.walking, self.last_walking, self.time_out, self.stair_final)
  def __repr__(self):
    return self.__str__()


# 60 seconds simulation
simulationTime = 120000
simulationInterval = 100 # 100 ms interval
# steps in stair
TOTAL_STAIRS = 30
# how many milliseconds does one new step appear?
SPEED = 500
WALKING = True
STAND_RATIO =0.5 # 0 for everyone is walking, 1 for everyone is standing

time = 0
cur_stair = 30

queue = []
out_list = []

# start simulation
while time < simulationTime:
#  print queue
  occupied = [x for x in queue if x.stair_final == cur_stair]
  # No one occupied this stair, then people can walk on this.
  if not occupied:
    rnd_walk = WALKING if random.random() > STAND_RATIO else not WALKING # add this to randomize the input(some one would stand still)
    person = Person(time, cur_stair, rnd_walk)
    queue.append(person)
#    print("1 person on step at time {}".format(time))
#    print queue



  # do simulation for each person on stairs
  for p in queue:
    # Check whether can walk: if p.stair_final - 1 and -2 are both empty, you can walk.
    can_walk = not [x for x in queue if x.stair_final == p.stair_final - 1]# and x.stair_final == p.stair_final - 2]
    result = p.simulation(time, can_walk, cur_stair - TOTAL_STAIRS)
    if not result: # simulation ends, so move to out_list
      queue.remove(p)
      out_list.append(p)
#      print "one person removed", p

#    print can_walk


  time += simulationInterval
  # Generate new stair if time is enough
  if time % SPEED == 0:
    cur_stair += 1

#    a = raw_input('......one step simulation, q to quit......')
#    if a == 'q':
#      break

hist_time = 0
# histogram for 10 seconds
hist_interval = 10000
while hist_time < simulationTime:
  out_part = [x for x in out_list if x.time_out >= hist_time and x.time_out < hist_time + hist_interval]
  avg_walking = sum([x.time_out - x.time_in for x in out_part])/(len(out_part) if out_part else 1)
  print("[{} ~ {}): out {} persons, avg on stair time {}".format(hist_time, hist_time+hist_interval, len(out_part), avg_walking))
  hist_time += hist_interval


print("queue has {} persons left".format(len(queue)))



