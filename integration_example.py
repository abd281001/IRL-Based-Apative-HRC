"""
Integration Example: Using Preference-Based Dataset Generator with Your IRL Setup

This script demonstrates how to integrate the preference-based dataset generator
with your existing IRL notebook and recipe generator.
"""

import sys
import json
import numpy as np
from preference_based_dataset_generator import (
    PreferenceBasedDatasetGenerator,
    PreferenceCategory,
    PreferenceConfig
)

# ===== STEP 1: Import or define your existing classes =====
# Copy your StateTracker and RecipeGenerator classes here
# For this example, I'll create minimal versions

class StateTracker:
    """Your existing StateTracker class"""
    def __init__(self):
        # Initialize as in your notebook
        ITEMS = ["pot", "pan", "plate", "tomato", "meat", "onion", "mushroom", "lettuce", "egg"]
        CUTTABLES = ["tomato", "onion", "mushroom", "lettuce"]
        LOCATIONS = ["A", "B", "C", "D", "E", "F"]
        
        self.feature_map = {}
        idx = 0
        
        # Location features
        for item in ITEMS:
            for loc in LOCATIONS:
                self.feature_map[f"{item}_{loc}"] = idx
                idx += 1
        
        # Cut status features
        for item in CUTTABLES:
            self.feature_map[f"{item}_cut"] = idx
            idx += 1
        
        # Washing status features
        for item in ITEMS:
            self.feature_map[f"{item}_washed"] = idx
            idx += 1
        
        # Global status features
        self.feature_map["stove_on"] = idx
        idx += 1
        self.feature_map["plate_served"] = idx
        
        self.n_features = idx + 1
        self.reset()
    
    def reset(self):
        """Reset state to initial conditions"""
        self.current_state = np.zeros(self.n_features, dtype=int)
        ITEMS = ["pot", "pan", "plate", "tomato", "meat", "onion", "mushroom", "lettuce", "egg"]
        for item in ITEMS:
            self.set_feature(f"{item}_A", 1)
    
    def set_feature(self, key, value):
        if key in self.feature_map:
            self.current_state[self.feature_map[key]] = value
    
    def get_state_vector(self):
        return self.current_state.copy()
    
    def apply_action(self, action_str):
        """Parse action string and update internal state vector"""
        if action_str.startswith("move"):
            content = action_str[action_str.find("(")+1 : action_str.find(")")].split()
            item, origin, _, dest = content
            self.set_feature(f"{item}_{origin}", 0)
            self.set_feature(f"{item}_{dest}", 1)
        elif action_str.startswith("cut"):
            content = action_str[action_str.find("(")+1 : action_str.find(")")].split()
            item = content.split()[0]
            self.set_feature(f"{item}_cut", 1)
        elif action_str.startswith("turn_on"):
            self.set_feature("stove_on", 1)
        elif action_str.startswith("turn_off"):
            self.set_feature("stove_on", 0)
        elif action_str.startswith("serve"):
            self.set_feature("plate_served", 1)
        elif action_str.startswith("wash"):
            content = action_str[action_str.find("(")+1 : action_str.find(")")].split()
            item = content[0]
            self.set_feature(f"{item}_washed", 1)


class RecipeGenerator:
    """Your existing RecipeGenerator class"""
    def __init__(self):
        self.demos = []
        self.tracker = StateTracker()
    
    def _record_trajectory(self, actions):
        """Run actions through state tracker and record (state, action) pairs"""
        self.tracker.reset()
        trajectory = []
        for action in actions:
            state_vector = self.tracker.get_state_vector().tolist()
            trajectory.append((state_vector, action))
            self.tracker.apply_action(action)
        
        final_state_vector = self.tracker.get_state_vector().tolist()
        trajectory.append((final_state_vector, "stop"))
        self.demos.append(trajectory)
    
    # Recipe Definitions
    def generate_grilled_steak(self):
        return [
            "move (pan A to C)", "move (meat A to C)", "turn_on (stove C)", 
            "move (plate A to B)", "turn_off (stove C)", "move (meat C to B)", 
            "move (plate B to D)", "move (plate D to E)", "serve (plate E)", 
            "move (pan C to F)", "wash (pan F)"
        ]
    
    def generate_salad(self):
        return [
            "move (lettuce A to B)", "cut (lettuce B)", "move (onion A to B)", 
            "cut (onion B)", "move (plate A to B)", "move (plate B to D)", 
            "move (plate D to E)", "serve (plate E)", "move (plate E to F)", 
            "wash (plate F)"
        ]
    
    def generate_burger(self):
        return [
            "move (pan A to C)", "move (meat A to C)", "turn_on (stove C)", 
            "move (lettuce A to B)", "cut (lettuce B)", "turn_off (stove C)", 
            "move (meat C to B)", "move (plate A to B)", "move (plate B to D)", 
            "move (plate D to E)", "serve (plate E)", "move (pan C to F)", 
            "wash (pan F)", "move (plate E to F)", "wash (plate F)"
        ]
    
    def generate_tomato_onion_soup(self):
        return [
            "move (pot A to C)", "move (tomato A to B)", "cut (tomato B)", 
            "move (tomato B to C)", "move (onion A to B)", "cut (onion B)", 
            "move (onion B to C)", "turn_on (stove C)", "move (plate A to D)", 
            "move (pot C to D)", "move (plate D to E)", "serve (plate E)", 
            "move (pot D to F)", "wash (pot F)"
        ]
    
    def generate_mushroom_soup(self):
        return [
            "move (pot A to C)", "move (mushroom A to B)", "cut (mushroom B)", 
            "move (mushroom B to C)", "move (onion A to B)", "cut (onion B)", 
            "move (onion B to C)", "turn_on (stove C)", "move (plate A to D)", 
            "move (pot C to D)", "move (plate D to E)", "serve (plate E)", 
            "move (pot D to F)", "wash (pot F)", "move (plate E to F)", 
            "wash (plate F)"
        ]
    
    def generate_dataset_from_recipes(self, recipe_functions, num_demos_per_recipe=10):
        """Generate demonstrations from a list of recipe functions"""
        import random
        print(f"Generating dataset from {len(recipe_functions)} recipes...")
        
        for recipe_func in recipe_functions:
            for _ in range(num_demos_per_recipe):
                actions = recipe_func()
                self._record_trajectory(actions)
        
        print(f"Generated {len(self.demos)} total demonstrations")
        return self.demos


# ===== STEP 2: Create workflow functions =====

def workflow_1_basic_preference_dataset():
    """
    Workflow 1: Generate basic preference-based dataset
    This is the simplest use case - generate datasets for all preference permutations
    """
    print("\n" + "="*60)
    print("WORKFLOW 1: Basic Preference Dataset Generation")
    print("="*60)
    
    # Step 1: Create your recipe generator and generate base demonstrations
    recipe_gen = RecipeGenerator()
    
    # Define which recipes to include (DYNAMIC - you can add/remove recipes here!)
    available_recipes = [
        recipe_gen.generate_grilled_steak,
        recipe_gen.generate_salad,
        recipe_gen.generate_burger,
    ]
    
    # Generate base demonstrations
    base_demos = recipe_gen.generate_dataset_from_recipes(
        available_recipes, 
        num_demos_per_recipe=5
    )
    
    # Step 2: Create preference-based generator
    pref_generator = PreferenceBasedDatasetGenerator(recipe_gen.tracker)
    pref_generator.set_base_demonstrations(base_demos)
    
    # Step 3: Generate all preference permutations
    # This creates 512 different datasets (2^9 preferences)
    all_datasets = pref_generator.generate_full_preference_dataset(
        output_dir="./preference_datasets_all",
        save_individual=True
    )
    
    print(f"\n✓ Generated {len(all_datasets)} preference-based agent populations")
    
    return all_datasets


def workflow_2_category_specific():
    """
    Workflow 2: Generate category-specific datasets
    Create separate datasets for each preference category (Plating, Cooking, Retrieval)
    Each category gets 8 agents (2^3 preferences)
    """
    print("\n" + "="*60)
    print("WORKFLOW 2: Category-Specific Dataset Generation")
    print("="*60)
    
    # Setup
    recipe_gen = RecipeGenerator()
    available_recipes = [
        recipe_gen.generate_tomato_onion_soup,
        recipe_gen.generate_mushroom_soup,
    ]
    base_demos = recipe_gen.generate_dataset_from_recipes(available_recipes, 5)
    
    pref_generator = PreferenceBasedDatasetGenerator(recipe_gen.tracker)
    pref_generator.set_base_demonstrations(base_demos)
    
    # Generate for each category
    categories = [
        PreferenceCategory.PLATING,
        PreferenceCategory.COOKING,
        PreferenceCategory.RETRIEVAL
    ]
    
    category_datasets = {}
    
    for category in categories:
        output_file = f"./preference_datasets_{category.value}.json"
        datasets = pref_generator.generate_category_specific_dataset(
            category, 
            output_file
        )
        category_datasets[category.value] = datasets
    
    print(f"\n✓ Generated category-specific datasets")
    print(f"  - Plating: {len(category_datasets['plating'])} agents")
    print(f"  - Cooking: {len(category_datasets['cooking'])} agents")
    print(f"  - Retrieval: {len(category_datasets['retrieval'])} agents")
    
    return category_datasets


def workflow_3_custom_preferences():
    """
    Workflow 3: Add custom preferences for your specific recipes
    This shows how to extend the system with new preferences
    """
    print("\n" + "="*60)
    print("WORKFLOW 3: Custom Preference Addition")
    print("="*60)
    
    # Setup
    recipe_gen = RecipeGenerator()
    pref_generator = PreferenceBasedDatasetGenerator(recipe_gen.tracker)
    
    # Add a custom preference
    def detect_stove_usage(state, action, next_state):
        """Custom detection: prefer turning on stove"""
        return action.startswith("turn_on")
    
    pref_generator.add_preference(
        "efficient_stove_usage",
        PreferenceCategory.CUSTOM,
        PreferenceConfig(
            name="Efficient Stove Usage",
            encouraged_reward=25.0,
            discouraged_reward=-10.0,
            detection_function=detect_stove_usage
        )
    )
    
    print("✓ Added custom preference: Efficient Stove Usage")
    print(f"✓ Total preferences now: {len(pref_generator.preferences)}")
    
    # Generate demos with custom preference
    available_recipes = [recipe_gen.generate_burger]
    base_demos = recipe_gen.generate_dataset_from_recipes(available_recipes, 3)
    pref_generator.set_base_demonstrations(base_demos)
    
    # Generate dataset with just this preference active
    config = {pid: False for pid in pref_generator.preferences.keys()}
    config["efficient_stove_usage"] = True
    
    custom_dataset = pref_generator.generate_dataset_with_preference(config)
    
    # Save
    with open("./custom_preference_dataset.json", 'w') as f:
        json.dump(custom_dataset, f, indent=2)
    
    print("✓ Generated dataset with custom preference")
    
    return custom_dataset


def workflow_4_dynamic_recipe_handling():
    """
    Workflow 4: Dynamic recipe addition/removal
    Shows how the system adapts when recipes are added or removed
    """
    print("\n" + "="*60)
    print("WORKFLOW 4: Dynamic Recipe Handling")
    print("="*60)
    
    recipe_gen = RecipeGenerator()
    pref_generator = PreferenceBasedDatasetGenerator(recipe_gen.tracker)
    
    # Initial recipe set
    print("\n--- Phase 1: Initial Recipe Set ---")
    recipes_phase1 = [
        recipe_gen.generate_salad,
        recipe_gen.generate_grilled_steak,
    ]
    demos_phase1 = recipe_gen.generate_dataset_from_recipes(recipes_phase1, 3)
    pref_generator.set_base_demonstrations(demos_phase1)
    
    # Generate initial preference dataset
    config = {"plating_ingredients": True, "washing_plates": True}
    dataset_phase1 = pref_generator.generate_dataset_with_preference(config)
    print(f"Generated {len(dataset_phase1)} demonstrations with 2 recipes")
    
    # Add more recipes
    print("\n--- Phase 2: Adding More Recipes ---")
    recipe_gen.demos = []  # Reset
    recipes_phase2 = [
        recipe_gen.generate_salad,
        recipe_gen.generate_grilled_steak,
        recipe_gen.generate_burger,  # NEW
        recipe_gen.generate_mushroom_soup,  # NEW
    ]
    demos_phase2 = recipe_gen.generate_dataset_from_recipes(recipes_phase2, 3)
    pref_generator.set_base_demonstrations(demos_phase2)
    
    # Generate with same preferences but more recipes
    dataset_phase2 = pref_generator.generate_dataset_with_preference(config)
    print(f"Generated {len(dataset_phase2)} demonstrations with 4 recipes")
    
    # Remove some recipes
    print("\n--- Phase 3: Removing Recipes ---")
    recipe_gen.demos = []  # Reset
    recipes_phase3 = [
        recipe_gen.generate_burger,
        recipe_gen.generate_mushroom_soup,
    ]
    demos_phase3 = recipe_gen.generate_dataset_from_recipes(recipes_phase3, 3)
    pref_generator.set_base_demonstrations(demos_phase3)
    
    dataset_phase3 = pref_generator.generate_dataset_with_preference(config)
    print(f"Generated {len(dataset_phase3)} demonstrations with 2 recipes (different set)")
    
    print("\n✓ Successfully demonstrated dynamic recipe handling!")
    print("  The preference system adapts automatically to any recipe set")
    
    return {
        "phase1": dataset_phase1,
        "phase2": dataset_phase2,
        "phase3": dataset_phase3
    }


def workflow_5_selective_preferences():
    """
    Workflow 5: Enable/disable specific preferences
    Useful for testing different preference combinations
    """
    print("\n" + "="*60)
    print("WORKFLOW 5: Selective Preference Testing")
    print("="*60)
    
    recipe_gen = RecipeGenerator()
    pref_generator = PreferenceBasedDatasetGenerator(recipe_gen.tracker)
    
    # Generate base data
    recipes = [recipe_gen.generate_tomato_onion_soup]
    base_demos = recipe_gen.generate_dataset_from_recipes(recipes, 5)
    pref_generator.set_base_demonstrations(base_demos)
    
    # Test different preference combinations
    test_configs = [
        {
            "name": "Only Plating Preferences",
            "config": {
                "plating_ingredients": True,
                "washing_plates": True,
                "delivering_dishes": True,
            }
        },
        {
            "name": "Only Cooking Preferences",
            "config": {
                "chopping_ingredients": True,
                "grilling_meat_mushroom": True,
            }
        },
        {
            "name": "Mixed Preferences",
            "config": {
                "chopping_ingredients": True,
                "plating_ingredients": True,
                "washing_plates": True,
            }
        },
    ]
    
    results = {}
    
    for test in test_configs:
        print(f"\n--- Testing: {test['name']} ---")
        
        # Set config (all False except specified)
        config = {pid: False for pid in pref_generator.preferences.keys()}
        config.update(test['config'])
        
        # Generate dataset
        dataset = pref_generator.generate_dataset_with_preference(config)
        
        # Analyze reward shaping
        total_shaping = sum(
            step['reward_shaping'] 
            for demo in dataset 
            for step in demo
        )
        
        print(f"  Total reward shaping: {total_shaping:.2f}")
        print(f"  Active preferences: {list(test['config'].keys())}")
        
        results[test['name']] = dataset
    
    print("\n✓ Tested multiple preference combinations")
    
    return results


# ===== MAIN EXECUTION =====

def main():
    """Run all workflows"""
    import os
    
    # Create output directories
    os.makedirs("./preference_datasets_all", exist_ok=True)
    
    print("\n" + "="*60)
    print("PREFERENCE-BASED DATASET GENERATOR - INTEGRATION DEMO")
    print("="*60)
    print("\nThis demo shows 5 different workflows for using the")
    print("preference-based dataset generator with your IRL setup.")
    print("="*60)
    
    # Run workflows (comment out ones you don't need)
    
    # Workflow 1: Basic - generate all 512 permutations
    # workflow_1_basic_preference_dataset()
    
    # Workflow 2: Category-specific - 8 agents per category
    workflow_2_category_specific()
    
    # Workflow 3: Custom preferences
    workflow_3_custom_preferences()
    
    # Workflow 4: Dynamic recipe handling
    workflow_4_dynamic_recipe_handling()
    
    # Workflow 5: Selective preference testing
    workflow_5_selective_preferences()
    
    print("\n" + "="*60)
    print("ALL WORKFLOWS COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
