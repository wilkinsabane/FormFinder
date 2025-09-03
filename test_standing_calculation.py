#!/usr/bin/env python3
"""Test Standing goal_difference calculation."""

from formfinder.DataFetcher import Standing

def test_standing_calculation():
    """Test that Standing objects calculate goal_difference correctly."""
    
    # Test case 1: Positive goal difference
    standing1 = Standing(
        position=1,
        team_id=123,
        team_name="Test Team 1",
        games_played=10,
        points=25,
        wins=8,
        draws=1,
        losses=1,
        goals_for=25,
        goals_against=8
    )
    
    print(f"Test 1 - Team: {standing1.team_name}")
    print(f"  Goals For: {standing1.goals_for}")
    print(f"  Goals Against: {standing1.goals_against}")
    print(f"  Goal Difference (property): {standing1.goal_difference}")
    print(f"  Expected: {standing1.goals_for - standing1.goals_against}")
    print(f"  Match: {standing1.goal_difference == (standing1.goals_for - standing1.goals_against)}")
    print()
    
    # Test case 2: Negative goal difference
    standing2 = Standing(
        position=15,
        team_id=456,
        team_name="Test Team 2",
        games_played=10,
        points=8,
        wins=2,
        draws=2,
        losses=6,
        goals_for=8,
        goals_against=18
    )
    
    print(f"Test 2 - Team: {standing2.team_name}")
    print(f"  Goals For: {standing2.goals_for}")
    print(f"  Goals Against: {standing2.goals_against}")
    print(f"  Goal Difference (property): {standing2.goal_difference}")
    print(f"  Expected: {standing2.goals_for - standing2.goals_against}")
    print(f"  Match: {standing2.goal_difference == (standing2.goals_for - standing2.goals_against)}")
    print()
    
    # Test case 3: Zero goal difference
    standing3 = Standing(
        position=10,
        team_id=789,
        team_name="Test Team 3",
        games_played=10,
        points=15,
        wins=4,
        draws=3,
        losses=3,
        goals_for=15,
        goals_against=15
    )
    
    print(f"Test 3 - Team: {standing3.team_name}")
    print(f"  Goals For: {standing3.goals_for}")
    print(f"  Goals Against: {standing3.goals_against}")
    print(f"  Goal Difference (property): {standing3.goal_difference}")
    print(f"  Expected: {standing3.goals_for - standing3.goals_against}")
    print(f"  Match: {standing3.goal_difference == (standing3.goals_for - standing3.goals_against)}")
    print()
    
    # Test all calculations are correct
    all_correct = (
        standing1.goal_difference == (standing1.goals_for - standing1.goals_against) and
        standing2.goal_difference == (standing2.goals_for - standing2.goals_against) and
        standing3.goal_difference == (standing3.goals_for - standing3.goals_against)
    )
    
    print(f"All calculations correct: {all_correct}")
    return all_correct

if __name__ == '__main__':
    test_standing_calculation()