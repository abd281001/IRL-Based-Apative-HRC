"""
Test Suite for Preference-Based Dataset Generator

Run this to verify the system works correctly with your setup.
"""

import numpy as np
import json
import os
from preference_based_dataset_generator import (
    PreferenceBasedDatasetGenerator,
    PreferenceCategory,
    PreferenceConfig,
    BehaviorPreference
)


def test_1_initialization():
    """Test 1: Basic initialization"""
    print("\n" + "="*60)
    print("TEST 1: Initialization")
    print("="*60)
    
    class DummyTracker:
        n_features = 69
    
    tracker = DummyTracker()
    gen = PreferenceBasedDatasetGenerator(tracker)
    
    # Check defaults
    assert len(gen.preferences) == 9, f"Expected 9 preferences, got {len(gen.preferences)}"
    
    # Check categories
    plating = sum(1 for p in gen.preferences.values() if p.category == PreferenceCategory.PLATING)
    cooking = sum(1 for p in gen.preferences.values() if p.category == PreferenceCategory.COOKING)
    retrieval = sum(1 for p in gen.preferences.values() if p.category == PreferenceCategory.RETRIEVAL)
    
    assert plating == 3, f"Expected 3 plating preferences, got {plating}"
    assert cooking == 3, f"Expected 3 cooking preferences, got {cooking}"
    assert retrieval == 3, f"Expected 3 retrieval preferences, got {retrieval}"
    
    print("✓ Initialization successful")
    print(f"✓ Created {len(gen.preferences)} default preferences")
    print(f"  - Plating: {plating}")
    print(f"  - Cooking: {cooking}")
    print(f"  - Retrieval: {retrieval}")


def test_2_preference_detection():
    """Test 2: Preference detection functions"""
    print("\n" + "="*60)
    print("TEST 2: Preference Detection")
    print("="*60)
    
    class DummyTracker:
        n_features = 69
    
    tracker = DummyTracker()
    gen = PreferenceBasedDatasetGenerator(tracker)
    
    # Test plating detection
    state = np.zeros(69)
    action_plating = "move (tomato B to D)"
    action_non_plating = "move (tomato B to C)"
    next_state = np.zeros(69)
    
    result_pos = gen._detects_plating_ingredient(state, action_plating, next_state)
    result_neg = gen._detects_plating_ingredient(state, action_non_plating, next_state)
    
    assert result_pos == True, "Should detect plating action"
    assert result_neg == False, "Should not detect non-plating action"
    
    # Test washing detection
    action_wash = "wash (plate F)"
    action_nowash = "move (plate D to E)"
    
    result_wash = gen._detects_washing_plate(state, action_wash, next_state)
    result_nowash = gen._detects_washing_plate(state, action_nowash, next_state)
    
    assert result_wash == True, "Should detect washing"
    assert result_nowash == False, "Should not detect non-washing"
    
    # Test chopping detection
    action_chop = "cut (onion B)"
    action_nochop = "move (onion B to C)"
    
    result_chop = gen._detects_chopping(state, action_chop, next_state)
    result_nochop = gen._detects_chopping(state, action_nochop, next_state)
    
    assert result_chop == True, "Should detect chopping"
    assert result_nochop == False, "Should not detect non-chopping"
    
    print("✓ All detection functions working correctly")
    print("  ✓ Plating detection")
    print("  ✓ Washing detection")
    print("  ✓ Chopping detection")


def test_3_reward_shaping():
    """Test 3: Reward shaping application"""
    print("\n" + "="*60)
    print("TEST 3: Reward Shaping")
    print("="*60)
    
    class DummyTracker:
        n_features = 69
    
    tracker = DummyTracker()
    gen = PreferenceBasedDatasetGenerator(tracker)
    
    state = np.zeros(69)
    next_state = np.zeros(69)
    
    # Test encouraged action
    action_encouraged = "move (tomato B to D)"  # Plating
    pref = gen.preferences["plating_ingredients"]
    
    reward = pref.apply_reward_shaping(state, action_encouraged, next_state)
    assert reward == 20.0, f"Expected +20, got {reward}"
    
    # Test discouraged action
    action_discouraged = "move (tomato B to C)"  # Not plating
    reward = pref.apply_reward_shaping(state, action_discouraged, next_state)
    assert reward == -30.0, f"Expected -30, got {reward}"
    
    # Test disabled preference
    pref.active = False
    reward = pref.apply_reward_shaping(state, action_encouraged, next_state)
    assert reward == 0.0, f"Expected 0 (disabled), got {reward}"
    
    print("✓ Reward shaping working correctly")
    print("  ✓ Encouraged actions: +20")
    print("  ✓ Discouraged actions: -30")
    print("  ✓ Disabled preferences: 0")


def test_4_permutation_generation():
    """Test 4: Preference permutation generation"""
    print("\n" + "="*60)
    print("TEST 4: Permutation Generation")
    print("="*60)
    
    class DummyTracker:
        n_features = 69
    
    tracker = DummyTracker()
    gen = PreferenceBasedDatasetGenerator(tracker)
    
    # Generate permutations
    perms = gen.generate_preference_permutations()
    
    # Should have 2^9 = 512 permutations
    assert len(perms) == 512, f"Expected 512 permutations, got {len(perms)}"
    
    # Check first and last
    all_false = all(not v for v in perms[0].values())
    all_true = all(v for v in perms[-1].values())
    
    assert all_false, "First permutation should have all preferences disabled"
    assert all_true, "Last permutation should have all preferences enabled"
    
    # Check uniqueness
    unique_perms = len(set(tuple(sorted(p.items())) for p in perms))
    assert unique_perms == 512, f"Expected 512 unique permutations, got {unique_perms}"
    
    print("✓ Permutation generation working correctly")
    print(f"  ✓ Generated {len(perms)} permutations")
    print("  ✓ All permutations unique")
    print("  ✓ Covers full preference space")


def test_5_dataset_generation():
    """Test 5: Dataset generation with preferences"""
    print("\n" + "="*60)
    print("TEST 5: Dataset Generation")
    print("="*60)
    
    class DummyTracker:
        n_features = 69
    
    tracker = DummyTracker()
    gen = PreferenceBasedDatasetGenerator(tracker)
    
    # Create dummy demonstrations
    demo1 = [
        (np.zeros(69).tolist(), "move (tomato A to B)"),
        (np.zeros(69).tolist(), "cut (tomato B)"),
        (np.zeros(69).tolist(), "move (tomato B to D)"),
        (np.zeros(69).tolist(), "serve (plate E)"),
        (np.zeros(69).tolist(), "stop")
    ]
    
    demo2 = [
        (np.zeros(69).tolist(), "move (plate A to D)"),
        (np.zeros(69).tolist(), "move (plate D to E)"),
        (np.zeros(69).tolist(), "serve (plate E)"),
        (np.zeros(69).tolist(), "wash (plate F)"),
        (np.zeros(69).tolist(), "stop")
    ]
    
    gen.set_base_demonstrations([demo1, demo2])
    
    # Generate with specific preferences
    config = {
        "plating_ingredients": True,
        "chopping_ingredients": True,
        "washing_plates": True,
    }
    
    dataset = gen.generate_dataset_with_preference(config)
    
    # Verify structure
    assert len(dataset) == 2, f"Expected 2 demonstrations, got {len(dataset)}"
    assert len(dataset[0]) == 5, f"Expected 5 steps in demo 1, got {len(dataset[0])}"
    
    # Verify reward shaping exists
    assert 'reward_shaping' in dataset[0][0], "Missing reward_shaping field"
    assert 'preference_details' in dataset[0][0], "Missing preference_details field"
    
    # Check that some reward shaping happened
    total_shaping = sum(step['reward_shaping'] for demo in dataset for step in demo)
    assert total_shaping != 0, "Expected non-zero reward shaping"
    
    print("✓ Dataset generation working correctly")
    print(f"  ✓ Generated {len(dataset)} demonstrations")
    print(f"  ✓ Total reward shaping: {total_shaping:.2f}")


def test_6_category_specific():
    """Test 6: Category-specific generation"""
    print("\n" + "="*60)
    print("TEST 6: Category-Specific Generation")
    print("="*60)
    
    class DummyTracker:
        n_features = 69
    
    tracker = DummyTracker()
    gen = PreferenceBasedDatasetGenerator(tracker)
    
    # Dummy demo
    demo = [(np.zeros(69).tolist(), "move (tomato A to D)"), 
            (np.zeros(69).tolist(), "stop")]
    gen.set_base_demonstrations([demo])
    
    # Test plating category
    output_file = "/tmp/test_plating.json"
    datasets = gen.generate_category_specific_dataset(
        PreferenceCategory.PLATING,
        output_file
    )
    
    # Should have 2^3 = 8 datasets
    assert len(datasets) == 8, f"Expected 8 datasets for plating, got {len(datasets)}"
    
    # Verify file was created
    assert os.path.exists(output_file), "Output file not created"
    
    # Load and verify
    with open(output_file) as f:
        loaded = json.load(f)
    assert len(loaded) == 8, "Loaded data doesn't match"
    
    # Cleanup
    os.remove(output_file)
    
    print("✓ Category-specific generation working correctly")
    print(f"  ✓ Generated {len(datasets)} datasets for PLATING category")


def test_7_custom_preferences():
    """Test 7: Adding custom preferences"""
    print("\n" + "="*60)
    print("TEST 7: Custom Preferences")
    print("="*60)
    
    class DummyTracker:
        n_features = 69
    
    tracker = DummyTracker()
    gen = PreferenceBasedDatasetGenerator(tracker)
    
    initial_count = len(gen.preferences)
    
    # Add custom preference
    def custom_detect(s, a, ns):
        return a.startswith("custom")
    
    gen.add_preference(
        "my_custom_pref",
        PreferenceCategory.CUSTOM,
        PreferenceConfig(
            name="My Custom",
            encouraged_reward=50.0,
            discouraged_reward=-10.0,
            detection_function=custom_detect
        )
    )
    
    assert len(gen.preferences) == initial_count + 1, "Preference not added"
    assert "my_custom_pref" in gen.preferences, "Custom preference not found"
    
    # Test it works
    state = np.zeros(69)
    next_state = np.zeros(69)
    
    pref = gen.preferences["my_custom_pref"]
    reward_pos = pref.apply_reward_shaping(state, "custom_action", next_state)
    reward_neg = pref.apply_reward_shaping(state, "other_action", next_state)
    
    assert reward_pos == 50.0, f"Expected +50, got {reward_pos}"
    assert reward_neg == -10.0, f"Expected -10, got {reward_neg}"
    
    # Test removal
    gen.remove_preference("my_custom_pref")
    assert len(gen.preferences) == initial_count, "Preference not removed"
    
    print("✓ Custom preferences working correctly")
    print("  ✓ Added custom preference")
    print("  ✓ Custom detection working")
    print("  ✓ Removed custom preference")


def test_8_dynamic_recipes():
    """Test 8: Dynamic recipe handling"""
    print("\n" + "="*60)
    print("TEST 8: Dynamic Recipe Handling")
    print("="*60)
    
    class DummyTracker:
        n_features = 69
    
    tracker = DummyTracker()
    gen = PreferenceBasedDatasetGenerator(tracker)
    
    # Phase 1: 2 recipes
    demos_phase1 = [
        [(np.zeros(69).tolist(), "move (A to B)"), (np.zeros(69).tolist(), "stop")],
        [(np.zeros(69).tolist(), "move (C to D)"), (np.zeros(69).tolist(), "stop")],
    ]
    gen.set_base_demonstrations(demos_phase1)
    
    config = {"plating_ingredients": True}
    dataset1 = gen.generate_dataset_with_preference(config)
    
    # Phase 2: 4 recipes
    demos_phase2 = demos_phase1 + [
        [(np.zeros(69).tolist(), "move (E to F)"), (np.zeros(69).tolist(), "stop")],
        [(np.zeros(69).tolist(), "move (G to H)"), (np.zeros(69).tolist(), "stop")],
    ]
    gen.set_base_demonstrations(demos_phase2)
    dataset2 = gen.generate_dataset_with_preference(config)
    
    # Phase 3: 1 recipe
    demos_phase3 = [demos_phase1[0]]
    gen.set_base_demonstrations(demos_phase3)
    dataset3 = gen.generate_dataset_with_preference(config)
    
    assert len(dataset1) == 2, f"Phase 1: Expected 2 demos, got {len(dataset1)}"
    assert len(dataset2) == 4, f"Phase 2: Expected 4 demos, got {len(dataset2)}"
    assert len(dataset3) == 1, f"Phase 3: Expected 1 demo, got {len(dataset3)}"
    
    print("✓ Dynamic recipe handling working correctly")
    print(f"  ✓ Phase 1 (2 recipes): {len(dataset1)} demonstrations")
    print(f"  ✓ Phase 2 (4 recipes): {len(dataset2)} demonstrations")
    print(f"  ✓ Phase 3 (1 recipe): {len(dataset3)} demonstrations")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print(" "*15 + "PREFERENCE-BASED DATASET GENERATOR")
    print(" "*25 + "TEST SUITE")
    print("="*70)
    
    tests = [
        test_1_initialization,
        test_2_preference_detection,
        test_3_reward_shaping,
        test_4_permutation_generation,
        test_5_dataset_generation,
        test_6_category_specific,
        test_7_custom_preferences,
        test_8_dynamic_recipes,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"\n✗ TEST FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n✗ TEST ERROR: {e}")
            failed += 1
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total Tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("\nYour preference-based dataset generator is ready to use!")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review the errors above.")
    
    print("="*70)


if __name__ == "__main__":
    run_all_tests()
