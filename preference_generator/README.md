# Preference-Based Dataset Generator for Adaptive IRL in HRC

A dynamic, extensible system for generating behavior preference (BP) datasets for Inverse Reinforcement Learning (IRL) in Human-Robot Collaboration (HRC) scenarios.

## Overview

This system implements the preference-based population training approach described in your reference image, where:
- **9 tunable preferences** are defined across 3 categories (Plating, Cooking, Retrieval)
- Each preference has **encouraged** (+reward) and **discouraged** (-penalty) reward shaping
- **2^N permutations** generate diverse agent populations (512 agents for 9 preferences)
- Agents are trained in self-play to create a diverse strategy space

### Key Features

✅ **Dynamic**: Works with any recipe that can be added or removed  
✅ **Flexible**: Preferences can be enabled/disabled individually  
✅ **Extensible**: Easy to add new preferences and categories  
✅ **Validated**: Ensures consistency across dataset  
✅ **Integration-Ready**: Works seamlessly with your existing IRL notebook

---

## Installation

```bash
# Copy the files to your project directory
cp preference_based_dataset_generator.py your_project/
cp integration_example.py your_project/

# Or clone if you have a repository
# git clone <repo_url>
```

**Requirements:**
- Python 3.7+
- NumPy
- JSON (standard library)

---

## Quick Start

### 1. Basic Usage

```python
from preference_based_dataset_generator import PreferenceBasedDatasetGenerator
from your_module import RecipeGenerator  # Your existing recipe generator

# Create your recipe generator
recipe_gen = RecipeGenerator()

# Define which recipes to use (DYNAMIC!)
recipes = [
    recipe_gen.generate_grilled_steak,
    recipe_gen.generate_salad,
    recipe_gen.generate_burger,
]

# Generate base demonstrations
recipe_gen.generate_dataset_from_recipes(recipes, num_demos_per_recipe=10)

# Create preference generator
pref_gen = PreferenceBasedDatasetGenerator(recipe_gen.tracker)
pref_gen.set_base_demonstrations(recipe_gen.demos)

# Generate all preference permutations (512 datasets)
all_datasets = pref_gen.generate_full_preference_dataset(
    output_dir="./preference_datasets",
    save_individual=True
)
```

### 2. Category-Specific Generation

Generate datasets for only one category (8 agents each):

```python
from preference_based_dataset_generator import PreferenceCategory

# Generate only for Plating category
plating_datasets = pref_gen.generate_category_specific_dataset(
    PreferenceCategory.PLATING,
    "plating_datasets.json"
)

# Similarly for COOKING or RETRIEVAL
```

---

## Preference System

### Default Preferences (from reference image)

| Category | Preference | Encouraged | Discouraged |
|----------|-----------|------------|-------------|
| **Plating** | Plating Ingredients | +20 | -30 |
| | Washing Plates | +20 | -30 |
| | Delivering Dishes | +20 | -30 |
| **Cooking** | Chopping Ingredients | +20 | -30 |
| | Potting Rice | +20 | -30 |
| | Grilling Meat/Mushroom | +20 | -30 |
| **Retrieval** | Taking Mushroom From Dispenser | +10 | -15 |
| | Taking Rice From Dispenser | +10 | -15 |
| | Taking Meat From Dispenser | +10 | -15 |

### Adding Custom Preferences

```python
from preference_based_dataset_generator import PreferenceConfig, PreferenceCategory

# Define detection function
def detect_my_preference(state, action, next_state):
    """Return True if action matches preference"""
    return action.startswith("my_action_type")

# Add to generator
pref_gen.add_preference(
    "my_custom_preference",
    PreferenceCategory.CUSTOM,
    PreferenceConfig(
        name="My Custom Preference",
        encouraged_reward=25.0,
        discouraged_reward=-20.0,
        detection_function=detect_my_preference
    )
)
```

---

## Dynamic Recipe Handling

The system automatically adapts when you add or remove recipes:

```python
# Initial set
recipes_v1 = [recipe_gen.generate_salad, recipe_gen.generate_burger]
demos_v1 = recipe_gen.generate_dataset_from_recipes(recipes_v1, 5)
pref_gen.set_base_demonstrations(demos_v1)
dataset_v1 = pref_gen.generate_full_preference_dataset()

# Add more recipes - system adapts automatically!
recipes_v2 = [
    recipe_gen.generate_salad,
    recipe_gen.generate_burger,
    recipe_gen.generate_tomato_soup,  # NEW
    recipe_gen.generate_mushroom_soup,  # NEW
]
recipe_gen.demos = []  # Reset
demos_v2 = recipe_gen.generate_dataset_from_recipes(recipes_v2, 5)
pref_gen.set_base_demonstrations(demos_v2)
dataset_v2 = pref_gen.generate_full_preference_dataset()

# Remove some recipes - still works!
recipes_v3 = [recipe_gen.generate_tomato_soup]
recipe_gen.demos = []
demos_v3 = recipe_gen.generate_dataset_from_recipes(recipes_v3, 5)
pref_gen.set_base_demonstrations(demos_v3)
dataset_v3 = pref_gen.generate_full_preference_dataset()
```

No code changes needed - just modify your recipe list!

---

## Workflows

### Workflow 1: Generate Full Population (512 agents)

```python
# Generate all 2^9 = 512 preference combinations
all_datasets = pref_gen.generate_full_preference_dataset(
    output_dir="./preference_datasets_all",
    save_individual=True  # Saves each agent's dataset separately
)

# Result: 512 JSON files, one per agent
# Format: agent_XXX_prefs_YYY.json
```

### Workflow 2: Category-Based Training

```python
# Train separate populations for each category
# 8 agents per category (2^3 preferences)

for category in [PreferenceCategory.PLATING, 
                 PreferenceCategory.COOKING, 
                 PreferenceCategory.RETRIEVAL]:
    datasets = pref_gen.generate_category_specific_dataset(
        category,
        f"datasets_{category.value}.json"
    )
```

### Workflow 3: Custom Preference Subset

```python
# Enable only specific preferences
config = {
    "plating_ingredients": True,
    "chopping_ingredients": True,
    "washing_plates": True,
    # All others default to False
}

dataset = pref_gen.generate_dataset_with_preference(config)
```

### Workflow 4: A/B Testing Preferences

```python
# Test different preference combinations
test_configs = [
    {"chopping_ingredients": True},
    {"plating_ingredients": True},
    {"chopping_ingredients": True, "plating_ingredients": True},
]

for i, config in enumerate(test_configs):
    dataset = pref_gen.generate_dataset_with_preference(config)
    # Analyze, compare, or use for training
```

---

## Output Format

### Individual Dataset File

```json
{
  "preference_config": {
    "plating_ingredients": true,
    "washing_plates": true,
    "chopping_ingredients": false,
    ...
  },
  "demonstrations": [
    [
      {
        "state": [0, 0, 1, ...],
        "action": "move (tomato A to B)",
        "reward_shaping": -30.0,
        "preference_details": {
          "plating_ingredients": -30.0
        }
      },
      ...
    ]
  ],
  "num_demonstrations": 15,
  "active_preferences": ["plating_ingredients", "washing_plates"]
}
```

### Summary File

```json
{
  "total_agents": 512,
  "base_demonstrations": 50,
  "preferences_available": [
    "plating_ingredients",
    "washing_plates",
    ...
  ],
  "dataset_names": [
    "agent_000_prefs_none",
    "agent_001_prefs_plating_ingredients",
    ...
  ]
}
```

---

## Integration with Your IRL Notebook

### Step 1: Generate Preference Datasets

```python
# In a new cell in your notebook
from preference_based_dataset_generator import PreferenceBasedDatasetGenerator

# Use your existing RecipeGenerator
gen = RecipeGenerator()
recipes = [gen.generate_tomato_onion_soup_1, gen.generate_grilled_steak]
gen.generate_random_dataset_from_recipes(recipes, 20)

# Create preference generator
pref_gen = PreferenceBasedDatasetGenerator(gen.tracker)
pref_gen.set_base_demonstrations(gen.demos)

# Generate datasets
all_datasets = pref_gen.generate_full_preference_dataset(
    output_dir="./bp_datasets"
)
```

### Step 2: Train IRL with Preference Data

```python
# Load a preference dataset
with open("./bp_datasets/agent_042_prefs_chopping.json") as f:
    agent_data = json.load(f)

# Convert to your IRL format
demonstrations = []
for demo in agent_data['demonstrations']:
    trajectory = [(tuple(step['state']), step['action']) for step in demo]
    demonstrations.append(trajectory)

# Run your existing IRL training
state_to_idx, idx_to_state, action_to_idx, idx_to_action = \
    create_state_action_mappings(demonstrations, unique_actions)

feature_matrix = extract_features(demonstrations, state_to_idx)

reward_weights, recovered_rewards = max_ent_irl(
    demonstrations, 
    feature_matrix, 
    state_to_idx, 
    action_to_idx
)
```

### Step 3: Self-Play Training (Population Method)

```python
# Train all 512 agents in self-play
import os

agents = []
for filename in os.listdir("./bp_datasets"):
    if filename.endswith(".json") and filename.startswith("agent_"):
        with open(f"./bp_datasets/{filename}") as f:
            agent_data = json.load(f)
        
        # Train this agent
        # ... (your IRL training code)
        
        agents.append({
            'id': filename,
            'preferences': agent_data['active_preferences'],
            'policy': trained_policy
        })

print(f"Trained {len(agents)} diverse agents!")
```

---

## Advanced Features

### 1. Selective Preference Testing

```python
# Test impact of individual preferences
for pref_id in pref_gen.preferences.keys():
    config = {pref_id: True}  # Only this preference active
    dataset = pref_gen.generate_dataset_with_preference(config)
    
    # Analyze reward shaping
    total_reward = sum(
        step['reward_shaping'] 
        for demo in dataset 
        for step in demo
    )
    print(f"{pref_id}: Total reward shaping = {total_reward}")
```

### 2. Progressive Difficulty

```python
# Start with simple recipes, gradually add complex ones
difficulty_levels = [
    [gen.generate_salad],  # Easy
    [gen.generate_salad, gen.generate_grilled_steak],  # Medium
    [gen.generate_salad, gen.generate_grilled_steak, 
     gen.generate_burger, gen.generate_mushroom_soup]  # Hard
]

for level, recipes in enumerate(difficulty_levels):
    demos = gen.generate_dataset_from_recipes(recipes, 10)
    pref_gen.set_base_demonstrations(demos)
    datasets = pref_gen.generate_full_preference_dataset(
        output_dir=f"./level_{level}_datasets"
    )
```

### 3. Preference Analysis

```python
# Analyze which preferences trigger most often
def analyze_preferences(dataset):
    pref_counts = {}
    
    for demo in dataset:
        for step in demo:
            for pref_id in step.get('preference_details', {}).keys():
                pref_counts[pref_id] = pref_counts.get(pref_id, 0) + 1
    
    return pref_counts

# Run analysis
config = {pid: True for pid in pref_gen.preferences.keys()}
dataset = pref_gen.generate_dataset_with_preference(config)
counts = analyze_preferences(dataset)

print("Preference Activation Frequency:")
for pref_id, count in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"  {pref_id}: {count} times")
```

---

## Troubleshooting

### Issue: Preferences not triggering

**Solution**: Check your detection functions. Add debug prints:

```python
def detect_plating(state, action, next_state):
    result = action.startswith("move") and "D" in action
    if result:
        print(f"MATCHED: {action}")
    return result
```

### Issue: Too many datasets (memory)

**Solution**: Generate in batches or use category-specific generation:

```python
# Instead of all 512 agents, generate by category
for category in [PreferenceCategory.PLATING, ...]:
    pref_gen.generate_category_specific_dataset(category, ...)
```

### Issue: Recipe-specific preferences not working

**Solution**: Create recipe-aware detection functions:

```python
def detect_in_specific_recipe(state, action, next_state):
    # Check if we're in a specific recipe context
    # by examining state features
    has_tomato = state[tomato_location_idx] > 0
    return has_tomato and action.startswith("cut")
```

---

## Architecture

```
PreferenceBasedDatasetGenerator
├── StateTracker (your existing class)
├── Preferences
│   ├── Plating Category (3 preferences)
│   ├── Cooking Category (3 preferences)
│   ├── Retrieval Category (3 preferences)
│   └── Custom Category (user-defined)
├── Base Demonstrations (from RecipeGenerator)
└── Output
    ├── Individual Agent Datasets (JSON)
    ├── Category-Specific Datasets (JSON)
    └── Summary Files (JSON)
```

---

## Citation

If using this in research, please cite the reference paper mentioned in your image:

```
Wang et al. [44] - Behavior Preference training approach with
diverse agent populations for HRC
```

---

## License

[Your License Here]

---

## Support

For issues, questions, or contributions, please [open an issue/contact].

---

## Changelog

### v1.0.0
- Initial release
- 9 default preferences across 3 categories
- Dynamic recipe handling
- Full permutation generation (512 agents)
- Category-specific generation
- Custom preference support
