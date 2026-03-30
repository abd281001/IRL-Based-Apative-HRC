My work is in Human-Robot collaboration. Here is an explanation of what I am trying to achieve: 


I have a task (recipes in this case) to be completed in collaboration with a robot. I initially train it (show it a demo for a single task (lets say generate_tomato_onion_soup_v1)) on a single recipe (as initialization). I see how well it predicts the future actions the next time it is online, observing that recipe. If however during testing (online run), I show it another demo (it will know its another recipe if it mispredicts multiple actions, we also assume the robot is manually (by the human before showing the demo) set to only observing when the human is performing a new recipe for the first time (not for existing recipe with different preferences)), I want it to store the new demo from its observations (online), then append that to what it originally learned and then retrain (when it goes offline) on the updated data (so it adapts to new preferences and/or recipes without forgetting what it previously learnt). 


as for how the system differentiates between the same recipe with diff preferences and a completely new recipe, the actions of the new demo can be compared with existing recipes in the dataset (jaccard distance). If there is a direct match, check for ordering. If its different ordering, its an existing recipe with updated preferences. If there were no direct matches with the jaccard distance, its a new recipe. 


I want the weight of the recipes decaying with time. If the dataset contains multiple instances of a recipe with multiple preferences, I want all the previous instances to decay with time. initially if a recipe is decayed in 10 time steps, and is removed from the training dataset, but the robot observes the user using the same recipe in 15 time steps, the system should adjust the decay rate to account for the fact that the user will use that recipe after a certain number of steps (note that for this a record has to be kept for when a recipe with a specific preference was added and when it was removed). This also links to the weight decay adjusting with an increase or decrease in the number of recipes because more recipes means a lower probability of a certain recipe being used. 


I want to do ordered testing as if simulating a real test run with the logic i have just described. For it to be systematic, I want to test different recipes with the same preference as the first recipe, with the same recipe with different preferences and different recipes with different preferences and see how well it performs before and after testing with a logical structured flow. 


I want the output to be easy to follow and to reflect the ability of the system to adapt to changes in recipes and preferences to show in a research paper at a conference such as HRI or ICRA. the goal of this work is to have a robot that adapts to a change in tasks or user preferences with time without catastrophic forgetting. If it sees any changes, it should retrain automatically and dynamically. I dont want anything hardcoded. First verify you have understood what i want you to do, then ill tell you whether or not to proceed.





Do you understand how this fits into the current setup?





keeping in mind what i have told you uptil now i.e., my task, my intended change of weighted recipes and linear weight decay and everything i have told you uptil now. Evaluate this file and tell me what improvements and optimizations you suggest.