

## Using data to cool data centers
Data centers consume 2–3% of worlds power¹. 30–50% of this power goes into keeping it cool². A system of different mechanisms works together to bring heat out from a datacenter and discard it into the atmosphere. These mechanims are controlled by their own local control systems. In this post, we detail how to control a system of systems more efficiently.

### Problem
**Why are they inefficient ?**
 - Local controls
 - Tacit knowledge
 - Complex interaction
 - Difficult to model

### Approach
**Can we design a better control system ?** 
 - Data based modelling
 - Fixed point optimisation 
 - Reinforcement learning on data model
 - Reinforcement learning directly on system
 - Continuous control

**Let us try this on a simple simulator ?** 
- Environment
	 - Red balls are hot, blue balls are cold
	 - Physics engine simulates motion of balls
	 - Reward is given when all servers have cooled down
	 - Time penalty for taking too long
	 - Pymunk engine
 - Trpo agent
 - Results

![Data center simulation](http://storage.googleapis.com/solveforx/therml/blog/images/dcvid.gif)

### Solution
**Modelling a real data center**
 - Sensory data from a real DC. Glance into data, simple EDA. 
 - Part based models
	 - Time delay in action
	 - LSTMs to simulate individual parts
	 - Each part connected to another
	 - Part connection graph
	 - State Machine composed of these parts is our simulator
	 - Controls, latent variables, 
	 - Accuracy of simulation
	 - Model sanity check
 - By product, predictive maintenance 

**Simple optimisation on data model**
 - Better setpoints according to weather
 - Reacting with a chiller instead of PAHUs
 
 ![PUE optimization](https://storage.googleapis.com/solveforx/therml/blog/images/pue_opt.png)
 
 **RL policies**
 - Action space of controls
 - Agent
 - Rewards
 - Results

### Taking to production
**System design**

 - Client side push
 - Time series database
 - Log cuts for model training
 - Model updates using dependency tree
 - What is a policy and how to deploy one ?
 - Monitoring
 - Fallback and safety mechanisms

### References
# dcool
data center cooling with reinforcement learning
