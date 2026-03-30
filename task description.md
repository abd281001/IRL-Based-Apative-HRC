The Learning Pipeline:

Offline Initialization — Train on a single demo recipe (e.g., generate_tomato_onion_soup_v1) as a starting point.
Online Observation — During a live run, the robot predicts the next action. A misprediction signals it's encountering something new (a new recipe or a preference change on an existing recipe).
Trigger Logic — The robot is manually flagged by the human to be in "observation mode" only when a genuinely new recipe is being demonstrated (not for re-runs of known recipes with known preferences).
Online Storage — It stores the newly observed recipe sequence during the run.
Offline Retraining — When it goes offline, it retrains on the combined old + new data, adapting without forgetting.


first check jaccard distance to determine whether recipe exists (preference invariant)
then when verified then check for preference existence in the dataset 
keep the decayed recipes to compare if those recipes reappear later on. then update the rate of decay (global variable) to keep that preference for those timesteps
remove the weight staying at 1 to keep one instance of each recipe
change weight decay variable with number of recipes (because decaying in 5 steps but we have 8 recipes means theyll all decay so make sure they decay in more than 8)
try to make this worked with over cooked











users preferences change over time so how well can we adapt to that without the user manually specifying it (reducing task load without sacrificing perofrmance and eff)
without collecting significant data (singleshot and continual)



one shot and then self-supervised learning (took picture of board, that part is self supervised as well as data collection)
avoid catastrophic forgetting while keeping selective forgetting (weight decay)