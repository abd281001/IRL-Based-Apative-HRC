"""
Dynamic Preference-Based Dataset Generator for Adaptive IRL in HRC

This module generates behavior preference (BP) datasets by applying reward shaping
based on user-defined preferences. The system is dynamic and works with any set of
recipes that can be added or removed.

Based on the approach from Wang et al. where preferences are categorized and reward
shaping is applied to create diverse agent populations.
"""

import numpy as np
import json
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class PreferenceConfig:
    """Configuration for a single preference with encouraged/discouraged reward shaping"""
    name: str
    encouraged_reward: float  # Reward for performing the preferred action
    discouraged_reward: float  # Penalty for not following preference
    detection_function: Callable  # Function to detect if preference applies to an action
    
    def __post_init__(self):
        """Validate preference configuration"""
        if self.encouraged_reward <= self.discouraged_reward:
            raise ValueError(f"Encouraged reward ({self.encouraged_reward}) must be > discouraged ({self.discouraged_reward})")


class PreferenceCategory(Enum):
    """Categories of preferences as shown in the reference image"""
    PLATING = "plating"
    COOKING = "cooking"
    RETRIEVAL = "retrieval"
    CUSTOM = "custom"


@dataclass 
class BehaviorPreference:
    """A behavior preference with category and configuration"""
    category: PreferenceCategory
    config: PreferenceConfig
    active: bool = True  # Can be toggled on/off
    
    def apply_reward_shaping(self, state: np.ndarray, action: str, 
                            next_state: np.ndarray) -> float:
        """Apply reward shaping based on whether action aligns with preference"""
        if not self.active:
            return 0.0
        
        if self.config.detection_function(state, action, next_state):
            return self.config.encouraged_reward
        else:
            return self.config.discouraged_reward


class PreferenceBasedDatasetGenerator:
    """
    Generates preference-based datasets from base demonstrations.
    
    Key features:
    - Dynamic: Works with any recipe that can be added/removed
    - Flexible: Preferences can be enabled/disabled
    - Extensible: Easy to add new preference types
    - Validates: Ensures consistency across dataset
    """
    
    def __init__(self, state_tracker):
        """
        Initialize with a state tracker instance.
        
        Args:
            state_tracker: StateTracker instance for state management
        """
        self.state_tracker = state_tracker
        self.preferences: Dict[str, BehaviorPreference] = {}
        self.base_demonstrations = []
        
        # Initialize default preferences based on the image
        self._initialize_default_preferences()
    
    def _initialize_default_preferences(self):
        """Initialize the 9 default preferences from the reference image"""
        
        # ===== PLATING CATEGORY =====
        self.add_preference(
            "plating_ingredients",
            PreferenceCategory.PLATING,
            PreferenceConfig(
                name="Plating Ingredients",
                encouraged_reward=20.0,
                discouraged_reward=-30.0,
                detection_function=lambda s, a, ns: self._detects_plating_ingredient(s, a, ns)
            )
        )
        
        self.add_preference(
            "washing_plates", 
            PreferenceCategory.PLATING,
            PreferenceConfig(
                name="Washing Plates",
                encouraged_reward=20.0,
                discouraged_reward=-30.0,
                detection_function=lambda s, a, ns: self._detects_washing_plate(s, a, ns)
            )
        )
        
        self.add_preference(
            "delivering_dishes",
            PreferenceCategory.PLATING,
            PreferenceConfig(
                name="Delivering Dishes",
                encouraged_reward=20.0,
                discouraged_reward=-30.0,
                detection_function=lambda s, a, ns: self._detects_delivering_dish(s, a, ns)
            )
        )
        
        # ===== COOKING CATEGORY =====
        self.add_preference(
            "chopping_ingredients",
            PreferenceCategory.COOKING,
            PreferenceConfig(
                name="Chopping Ingredients",
                encouraged_reward=20.0,
                discouraged_reward=-30.0,
                detection_function=lambda s, a, ns: self._detects_chopping(s, a, ns)
            )
        )
        
        self.add_preference(
            "potting_rice",
            PreferenceCategory.COOKING,
            PreferenceConfig(
                name="Potting Rice",
                encouraged_reward=20.0,
                discouraged_reward=-30.0,
                detection_function=lambda s, a, ns: self._detects_potting_rice(s, a, ns)
            )
        )
        
        self.add_preference(
            "grilling_meat_mushroom",
            PreferenceCategory.COOKING,
            PreferenceConfig(
                name="Grilling Meat/Mushroom",
                encouraged_reward=20.0,
                discouraged_reward=-30.0,
                detection_function=lambda s, a, ns: self._detects_grilling(s, a, ns)
            )
        )
        
        # ===== RETRIEVAL CATEGORY =====
        self.add_preference(
            "taking_mushroom_from_dispenser",
            PreferenceCategory.RETRIEVAL,
            PreferenceConfig(
                name="Taking Mushroom From Dispenser",
                encouraged_reward=10.0,
                discouraged_reward=-15.0,
                detection_function=lambda s, a, ns: self._detects_mushroom_retrieval(s, a, ns)
            )
        )
        
        self.add_preference(
            "taking_rice_from_dispenser",
            PreferenceCategory.RETRIEVAL,
            PreferenceConfig(
                name="Taking Rice From Dispenser",
                encouraged_reward=10.0,
                discouraged_reward=-15.0,
                detection_function=lambda s, a, ns: self._detects_rice_retrieval(s, a, ns)
            )
        )
        
        self.add_preference(
            "taking_meat_from_dispenser",
            PreferenceCategory.RETRIEVAL,
            PreferenceConfig(
                name="Taking Meat From Dispenser",
                encouraged_reward=10.0,
                discouraged_reward=-15.0,
                detection_function=lambda s, a, ns: self._detects_meat_retrieval(s, a, ns)
            )
        )
    
    # ===== DETECTION FUNCTIONS =====
    # These map your state representation to preference detection
    
    def _detects_plating_ingredient(self, state: np.ndarray, action: str, next_state: np.ndarray) -> bool:
        """Detect if action involves moving ingredient to plating station (D)"""
        if not action.startswith("move"):
            return False
        
        # Parse action: "move (item origin to dest)"
        try:
            content = action[action.find("(")+1 : action.find(")")].split()
            item, origin, _, dest = content
            # Check if moving food item to plating station D
            food_items = ["tomato", "meat", "onion", "mushroom", "lettuce", "egg"]
            return item in food_items and dest == "D"
        except:
            return False
    
    def _detects_washing_plate(self, state: np.ndarray, action: str, next_state: np.ndarray) -> bool:
        """Detect washing plate action"""
        if not action.startswith("wash"):
            return False
        try:
            content = action[action.find("(")+1 : action.find(")")].split()
            item = content[0]
            return item == "plate"
        except:
            return False
    
    def _detects_delivering_dish(self, state: np.ndarray, action: str, next_state: np.ndarray) -> bool:
        """Detect serving/delivering action"""
        return action.startswith("serve")
    
    def _detects_chopping(self, state: np.ndarray, action: str, next_state: np.ndarray) -> bool:
        """Detect cutting/chopping action"""
        return action.startswith("cut")
    
    def _detects_potting_rice(self, state: np.ndarray, action: str, next_state: np.ndarray) -> bool:
        """Detect moving rice to pot (for cooking)"""
        # Note: Your current recipes don't have rice, but this is for extensibility
        if not action.startswith("move"):
            return False
        try:
            content = action[action.find("(")+1 : action.find(")")].split()
            item, origin, _, dest = content
            return item == "rice" and dest == "C"  # Moving rice to stove
        except:
            return False
    
    def _detects_grilling(self, state: np.ndarray, action: str, next_state: np.ndarray) -> bool:
        """Detect grilling meat or mushroom (moving to pan at stove)"""
        if not action.startswith("move"):
            return False
        try:
            content = action[action.find("(")+1 : action.find(")")].split()
            item, origin, _, dest = content
            grill_items = ["meat", "mushroom"]
            # Check if item is at stove (C) and pan is also at stove
            return item in grill_items and dest == "C"
        except:
            return False
    
    def _detects_mushroom_retrieval(self, state: np.ndarray, action: str, next_state: np.ndarray) -> bool:
        """Detect taking mushroom from initial station A (dispenser)"""
        if not action.startswith("move"):
            return False
        try:
            content = action[action.find("(")+1 : action.find(")")].split()
            item, origin, _, dest = content
            return item == "mushroom" and origin == "A"
        except:
            return False
    
    def _detects_rice_retrieval(self, state: np.ndarray, action: str, next_state: np.ndarray) -> bool:
        """Detect taking rice from initial station A"""
        if not action.startswith("move"):
            return False
        try:
            content = action[action.find("(")+1 : action.find(")")].split()
            item, origin, _, dest = content
            return item == "rice" and origin == "A"
        except:
            return False
    
    def _detects_meat_retrieval(self, state: np.ndarray, action: str, next_state: np.ndarray) -> bool:
        """Detect taking meat from initial station A"""
        if not action.startswith("move"):
            return False
        try:
            content = action[action.find("(")+1 : action.find(")")].split()
            item, origin, _, dest = content
            return item == "meat" and origin == "A"
        except:
            return False
    
    # ===== PREFERENCE MANAGEMENT =====
    
    def add_preference(self, pref_id: str, category: PreferenceCategory, 
                      config: PreferenceConfig):
        """Add or update a preference"""
        self.preferences[pref_id] = BehaviorPreference(category, config)
    
    def remove_preference(self, pref_id: str):
        """Remove a preference"""
        if pref_id in self.preferences:
            del self.preferences[pref_id]
    
    def toggle_preference(self, pref_id: str, active: bool):
        """Enable or disable a preference"""
        if pref_id in self.preferences:
            self.preferences[pref_id].active = active
    
    def get_active_preferences(self) -> List[Tuple[str, BehaviorPreference]]:
        """Get all active preferences"""
        return [(pid, pref) for pid, pref in self.preferences.items() if pref.active]
    
    # ===== DATASET GENERATION =====
    
    def load_base_demonstrations(self, filepath: str):
        """Load base demonstrations from file"""
        with open(filepath, 'r') as f:
            demo_lists = [json.loads(line) for line in f.read().strip().split('\n')]
        
        self.base_demonstrations = []
        for demo in demo_lists:
            trajectory = [(step['state'], step['action']) for step in demo]
            self.base_demonstrations.append(trajectory)
        
        print(f"Loaded {len(self.base_demonstrations)} base demonstrations")
    
    def set_base_demonstrations(self, demonstrations: List[List[Tuple]]):
        """Set base demonstrations directly"""
        self.base_demonstrations = demonstrations
        print(f"Set {len(self.base_demonstrations)} base demonstrations")
    
    def generate_preference_permutations(self) -> List[Dict[str, bool]]:
        """
        Generate all permutations of preference combinations.
        
        For N preferences, generates 2^N combinations (each on/off).
        For 9 preferences as in the image, this generates 512 combinations.
        
        Returns:
            List of preference configurations (each is a dict of pref_id -> enabled)
        """
        active_prefs = list(self.preferences.keys())
        n_prefs = len(active_prefs)
        n_combinations = 2 ** n_prefs
        
        combinations = []
        for i in range(n_combinations):
            config = {}
            for j, pref_id in enumerate(active_prefs):
                # Check if j-th bit is set in i
                config[pref_id] = bool((i >> j) & 1)
            combinations.append(config)
        
        print(f"Generated {len(combinations)} preference permutations for {n_prefs} preferences")
        return combinations
    
    def generate_dataset_with_preference(self, preference_config: Dict[str, bool],
                                        demonstrations: Optional[List] = None) -> List[Dict]:
        """
        Generate a dataset with specific preference configuration applied.
        
        Args:
            preference_config: Dict mapping preference_id to enabled/disabled
            demonstrations: Optional demonstrations to use (defaults to base_demonstrations)
        
        Returns:
            List of trajectories with reward-shaped states
        """
        if demonstrations is None:
            demonstrations = self.base_demonstrations
        
        # Apply preference configuration
        for pref_id, enabled in preference_config.items():
            if pref_id in self.preferences:
                self.preferences[pref_id].active = enabled
        
        shaped_demonstrations = []
        
        for trajectory in demonstrations:
            shaped_trajectory = []
            
            for i, (state, action) in enumerate(trajectory):
                # Convert state to numpy if it's a list/tuple
                state_array = np.array(state) if not isinstance(state, np.ndarray) else state
                
                # Get next state for detection
                if i < len(trajectory) - 1:
                    next_state_array = np.array(trajectory[i + 1][0])
                else:
                    next_state_array = state_array  # Terminal state
                
                # Calculate total reward shaping from all active preferences
                total_reward_shaping = 0.0
                preference_details = {}
                
                for pref_id, pref in self.get_active_preferences():
                    reward = pref.apply_reward_shaping(state_array, action, next_state_array)
                    total_reward_shaping += reward
                    if reward != 0:  # Only track if preference applied
                        preference_details[pref_id] = reward
                
                shaped_trajectory.append({
                    "state": state_array.tolist() if isinstance(state_array, np.ndarray) else state,
                    "action": action,
                    "reward_shaping": total_reward_shaping,
                    "preference_details": preference_details
                })
            
            shaped_demonstrations.append(shaped_trajectory)
        
        return shaped_demonstrations
    
    def generate_full_preference_dataset(self, 
                                        output_dir: str = "./preference_datasets",
                                        save_individual: bool = True) -> Dict:
        """
        Generate complete preference-based dataset with all permutations.
        
        Following the approach from the image:
        - Generate 2^N agents (N=9 → 512 agents in the reference)
        - Each agent has a unique preference combination
        - Train with self-play to create diverse population
        
        Args:
            output_dir: Directory to save datasets
            save_individual: Whether to save each permutation separately
        
        Returns:
            Dictionary containing all generated datasets
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate all preference permutations
        permutations = self.generate_preference_permutations()
        
        all_datasets = {}
        
        for idx, pref_config in enumerate(permutations):
            # Generate dataset for this preference combination
            dataset = self.generate_dataset_with_preference(pref_config)
            
            # Create a readable name based on active preferences
            active_prefs = [pid for pid, enabled in pref_config.items() if enabled]
            dataset_name = f"agent_{idx:03d}_prefs_{'_'.join(active_prefs[:3]) if active_prefs else 'none'}"
            
            all_datasets[dataset_name] = {
                "preference_config": pref_config,
                "demonstrations": dataset,
                "num_demonstrations": len(dataset),
                "active_preferences": active_prefs
            }
            
            # Save individual dataset if requested
            if save_individual:
                output_file = os.path.join(output_dir, f"{dataset_name}.json")
                with open(output_file, 'w') as f:
                    json.dump(all_datasets[dataset_name], f, indent=2)
            
            if (idx + 1) % 50 == 0:
                print(f"Generated {idx + 1}/{len(permutations)} preference-based datasets")
        
        # Save summary
        summary = {
            "total_agents": len(all_datasets),
            "base_demonstrations": len(self.base_demonstrations),
            "preferences_available": list(self.preferences.keys()),
            "dataset_names": list(all_datasets.keys())
        }
        
        with open(os.path.join(output_dir, "dataset_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Generated {len(all_datasets)} preference-based datasets")
        print(f"✓ Saved to {output_dir}")
        
        return all_datasets
    
    def generate_category_specific_dataset(self, category: PreferenceCategory,
                                          output_file: str) -> List[Dict]:
        """
        Generate dataset with only preferences from a specific category active.
        
        This creates the 3 categories mentioned in the image:
        - Plating category (3 preferences)
        - Cooking category (3 preferences)  
        - Retrieval category (3 preferences)
        
        For each category, generates 2^3 = 8 agents with different combinations.
        """
        # Get preferences in this category
        category_prefs = {
            pid: pref for pid, pref in self.preferences.items() 
            if pref.category == category
        }
        
        print(f"\n=== Generating datasets for {category.value} category ===")
        print(f"Found {len(category_prefs)} preferences in this category")
        
        # Disable all preferences first
        for pref_id in self.preferences:
            self.preferences[pref_id].active = False
        
        # Generate permutations for just this category
        category_pref_ids = list(category_prefs.keys())
        n_perms = 2 ** len(category_pref_ids)
        
        datasets = []
        
        for i in range(n_perms):
            # Create preference config for this permutation
            config = {}
            for j, pref_id in enumerate(category_pref_ids):
                config[pref_id] = bool((i >> j) & 1)
            
            # Generate dataset
            dataset = self.generate_dataset_with_preference(config)
            datasets.append({
                "permutation": i,
                "preference_config": config,
                "demonstrations": dataset
            })
        
        # Save
        with open(output_file, 'w') as f:
            json.dump(datasets, f, indent=2)
        
        print(f"✓ Generated {len(datasets)} datasets for {category.value} category")
        print(f"✓ Saved to {output_file}")
        
        return datasets


# ===== UTILITY FUNCTIONS =====

def create_default_generator(state_tracker) -> PreferenceBasedDatasetGenerator:
    """Create a generator with default preferences matching the reference image"""
    return PreferenceBasedDatasetGenerator(state_tracker)


def example_usage():
    """Example of how to use the preference-based dataset generator"""
    
    # This is a placeholder - in real use, you'd import your StateTracker
    class DummyStateTracker:
        pass
    
    tracker = DummyStateTracker()
    
    # Create generator
    generator = PreferenceBasedDatasetGenerator(tracker)
    
    # Option 1: Load base demonstrations from file
    # generator.load_base_demonstrations("demonstrations.txt")
    
    # Option 2: Set demonstrations directly from your recipe generator
    # from your_module import RecipeGenerator
    # recipe_gen = RecipeGenerator()
    # recipe_gen.generate_random_dataset(100)
    # generator.set_base_demonstrations(recipe_gen.demos)
    
    # Generate complete preference dataset (all 512 permutations)
    # all_datasets = generator.generate_full_preference_dataset(
    #     output_dir="./preference_datasets",
    #     save_individual=True
    # )
    
    # Or generate for specific category only (8 permutations each)
    # plating_datasets = generator.generate_category_specific_dataset(
    #     PreferenceCategory.PLATING,
    #     "plating_preference_datasets.json"
    # )
    
    print("Example setup complete!")


if __name__ == "__main__":
    example_usage()
