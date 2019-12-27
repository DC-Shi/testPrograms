# Escalator Simulator
I have question: Does it get a higher throughput when people trying to walk on running escalators?

So we want to create one to simulate escalator.

# Conditions and modeling
- People get lower latency when walking on escalator
- Walking needs at least 2 spaces between persons, while standing still just 1 or 0 steps (0 means each step stands one person

# Formulation
To formulate this process, we simplify the model:
- From time 0, we started to generate stairs(with increment index)
- The escalator has ```SPEED```, ```TOTAL_STAIRS``` as parameters
- For given time T, we can calculate exposed index interval [```start_T_i```, ```end_T_i```]
- Walking speed is 80 steps/min
- Walking is only available when current stair in in [```start_T_i```, ```end_T_i```], and has limitation of at least 2 spaces.
- For exiting point, no limitation, means multiple person can stand in same stair.
- Entry point always has people. Only one person allowed for one stairs at one time.

For each person, it has these attributes:
- Time stepping on stair
- Stair index
- Walking or not(true means if available spaces ahead, then walking)
- Last walking time
- Time stepping out stair
- Final stair index

# About simulation
We use microsecond as simulation level.

# Conclusion
We have nearly the same throughput, about 2 persons/sec, no matter person standing or walking on steps, under the assumption that there are many persons want to take escalator. 
But lantency for each person would be different, walking is less than standing.

If left walking and right standing is applied, it would not reach maximum utility since many people would wait for standing queue and left-walking-lane is less utilized.

So, from escalator's aspect, make sure each step holds two person.

From person's aspect, do it as you wish, walking if in hurry, but keep safety in mind.




Copyright (C) 2018 Daochen Shi