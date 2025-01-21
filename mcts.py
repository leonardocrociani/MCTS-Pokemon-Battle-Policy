from typing import List
from copy import deepcopy
import math
import numpy as np
import random
from tqdm import tqdm

from vgc.behaviour import BattlePolicy
from vgc.datatypes.Objects import GameState, PkmStatus, Pkm, PkmTeam, PkmType, WeatherCondition, PkmMove, PkmStat
from vgc.datatypes.Constants import DEFAULT_N_ACTIONS, TYPE_CHART_MULTIPLIER

SWITCH_THRESHOLD = 3.75
DAMAGE_THRESHOLD = 0.2

def n_fainted(t: PkmTeam): # from [@thunder battle policy] 
    fainted = 0
    fainted += t.active.hp == 0
    if len(t.party) > 0:
        fainted += t.party[0].hp == 0
    if len(t.party) > 1:
        fainted += t.party[1].hp == 0
    return fainted

def estimate_damage(move: PkmMove, pkm_type: PkmType, opp_pkm_type: PkmType,
                    attack_stage: int, defense_stage: int, weather: WeatherCondition) -> float: # from [@thunder battle policy]
    move_type: PkmType = move.type
    move_power: float = move.power
    type_rate = TYPE_CHART_MULTIPLIER[move_type][opp_pkm_type]
    if type_rate == 0:
        return 0
    if move.fixed_damage > 0:
        return move.fixed_damage
    stab = 1.5 if move_type == pkm_type else 1.
    if (move_type == PkmType.WATER and weather == WeatherCondition.RAIN) or (
            move_type == PkmType.FIRE and weather == WeatherCondition.SUNNY):
        weather = 1.5
    elif (move_type == PkmType.WATER and weather == WeatherCondition.SUNNY) or (
            move_type == PkmType.FIRE and weather == WeatherCondition.RAIN):
        weather = .5
    else:
        weather = 1.
    stage_level = attack_stage - defense_stage
    stage = (stage_level + 2.) / 2 if stage_level >= 0. else 2. / \
        (np.abs(stage_level) + 2.)
    damage = type_rate * \
        stab * weather * stage * move_power
    return damage


def is_action_suboptimal(game_state: GameState, action: int, effectiveness_threshold: float = 0.5) -> bool:
    active_moves = game_state.teams[0].active.moves

    # check if the action is a switch
    if action >= len(active_moves): 
        switch_index = action - len(active_moves)
        my_team = game_state.teams[0]
        
        if switch_index >= len(my_team.party) or my_team.party[switch_index].hp <= 0:
            return True  # avoid switching to a fainted Pokémon

        # let's eval the matchup of the current pokemon and of the switch pokemon
        current_matchup = evaluate_actives_matchup(
            my_team.active,
            game_state.teams[1].active
        )

        switch_matchup = evaluate_actives_matchup(
            my_team.party[switch_index],
            game_state.teams[1].active
        )

        # if the switch doesn't improve the matchup by a threshold, consider it suboptimal
        is_switch_suboptimal = (switch_matchup - SWITCH_THRESHOLD) < current_matchup
        return is_switch_suboptimal

    # for the attack moves let's compute the expected damage with `estimate_damage()`
    move = active_moves[action]
    attacker = game_state.teams[0].active
    defender = game_state.teams[1].active
    
    my_team = game_state.teams[0]
    my_attack_stage = my_team.stage[PkmStat.ATTACK]

    opp_team = game_state.teams[1]
    opp_defense_stage = opp_team.stage[PkmStat.DEFENSE]

    estimated_dmg = estimate_damage(
        move=move,
        pkm_type=attacker.type,
        opp_pkm_type=defender.type,
        attack_stage=my_attack_stage,
        defense_stage=opp_defense_stage,
        weather=game_state.weather
    )

    # if the damage is too low, i'll consider the move suboptimal
    defender_max_hp = defender.max_hp
    damage_ratio = estimated_dmg / defender_max_hp  # percentage of removed hp

    is_suboptimal = damage_ratio < DAMAGE_THRESHOLD or estimated_dmg <= effectiveness_threshold
    return is_suboptimal

def evaluate_actives_matchup(my_active: Pkm, opp_active:Pkm) -> float:

    my_active_type = my_active.type
    opp_active_type = opp_active.type
    my_moves_type = [m.type for m in my_active.moves]
    opp_moves_type = [m.type for m in opp_active.moves if m.name]
    
    # calculate base defensive matchup considering STAB
    defensive_match_up = 0.
    for mtype in opp_moves_type:
        multiplier = TYPE_CHART_MULTIPLIER[mtype][my_active_type]
        # apply STAB bonus if move type matches attacker's type
        if mtype == opp_active_type:
            multiplier *= 1.5
        defensive_match_up = max(multiplier, defensive_match_up)

    # calculate base offensive matchup considering STAB
    offensive_match_up = 0.
    for mtype in my_moves_type:
        multiplier = TYPE_CHART_MULTIPLIER[mtype][my_active_type]
        # apply STAB bonus if move type matches attacker's type
        if mtype == opp_active_type:
            multiplier *= 1.5
        offensive_match_up = max(multiplier, offensive_match_up)

    # calculate type coverage scores
    my_coverage_score = 0
    opp_coverage_score = 0
    
    # evaluate how many different types we can hit effectively
    type_coverage = set()
    for mtype in my_moves_type:
        for target_type in PkmType:
            if TYPE_CHART_MULTIPLIER[mtype][target_type] > 1:
                type_coverage.add(target_type)
    my_coverage_score = len(type_coverage) / len(PkmType)

    # evaluate opponent's type coverage
    opp_type_coverage = set()
    for mtype in opp_moves_type:
        for target_type in PkmType:
            if TYPE_CHART_MULTIPLIER[mtype][target_type] > 1:
                opp_type_coverage.add(target_type)
    opp_coverage_score = len(opp_type_coverage) / len(PkmType)

    # calculate immunities and resistances
    immunity_bonus = 0
    for mtype in opp_moves_type:
        if TYPE_CHART_MULTIPLIER[mtype][my_active_type] == 0:
            immunity_bonus += 0.5
        elif TYPE_CHART_MULTIPLIER[mtype][my_active_type] <= 0.5:
            immunity_bonus += 0.25

    # combine all factors
    matchup_score = (offensive_match_up * 1.2  # Weighted more towards offense 😡
                    - defensive_match_up 
                    + my_coverage_score * 0.5
                    - opp_coverage_score * 0.3
                    + immunity_bonus)

    return matchup_score

def detailed_status_eval(pkm: Pkm) -> float:
    if pkm.status == PkmStatus.PARALYZED:
        return -1.2  # severe speed penalty and chance to not move
    elif pkm.status == PkmStatus.SLEEP:
        return -1.5  # cannot move but temporary
    elif pkm.status == PkmStatus.FROZEN:
        return -1.8  # cannot move and rare to thaw
    elif pkm.status == PkmStatus.CONFUSED:
        return -0.8  # temporary but dangerous
    elif pkm.status == PkmStatus.BURNED:
        return -0.9  # attack reduction and damage
    elif pkm.status == PkmStatus.POISONED:
        return -0.7  # just damage
    return 0

def evaluate_stages(team: PkmTeam) -> float:
    
    stage_score = 0
    
    def update_stage_score(stage: any, weight: float) -> float:
        if stage > 0:
            return weight * (1 - (0.9 ** stage))
        else:
            return -weight * (1 - (0.9 ** abs(stage)))
            
    attack_stage = team.stage[PkmStat.ATTACK]
    defense_stage = team.stage[PkmStat.DEFENSE]
    speed_stage = team.stage[PkmStat.SPEED]
    
    stage_score += update_stage_score(attack_stage, 1.2)
    stage_score += update_stage_score(defense_stage, 1.0)
    stage_score += update_stage_score(speed_stage, 1.1)

    return stage_score
    
def evaluate_terminal_state(state: GameState, cycles: int, num_switches: int) -> float:
    my_team = state.teams[0]
    opp_team = state.teams[1]
    my_active: Pkm = my_team.active
    opp_active: Pkm = opp_team.active

    # base matchup evaluation
    match_up = evaluate_actives_matchup(
        my_active, 
        opp_active
    )

    # hp evaluation
    my_hp_ratio = my_active.hp / my_active.max_hp
    opp_hp_ratio = opp_active.hp / opp_active.max_hp
    
    # hp is more valuable when it's lower
    hp_weight = 4.0
    my_hp_score = hp_weight * (1 + (1 - my_hp_ratio)) * my_hp_ratio
    opp_hp_score = hp_weight * (1 + (1 - opp_hp_ratio)) * opp_hp_ratio

    # team health evaluation
    my_team_health = 0
    for pokemon in [my_team.party[0], my_team.party[1]]:
        if pokemon.hp > 0:
            health_ratio = pokemon.hp / pokemon.max_hp
            # backup pokemon health is important
            my_team_health += health_ratio * 2  

    my_status_score = detailed_status_eval(my_active)
    opp_status_score = detailed_status_eval(opp_active)

    my_stage_score = evaluate_stages(my_team)
    opp_stage_score = evaluate_stages(opp_team)


    # weather
    weather_score = 0
    if state.weather != WeatherCondition.CLEAR:
        if (state.weather == WeatherCondition.SUNNY and my_active.type == PkmType.FIRE) or \
           (state.weather == WeatherCondition.RAIN and my_active.type == PkmType.WATER):
            weather_score += 0.5
        elif (state.weather == WeatherCondition.SUNNY and opp_active.type == PkmType.FIRE) or \
             (state.weather == WeatherCondition.RAIN and opp_active.type == PkmType.WATER):
            weather_score -= 0.5

    # switching penalty
    switch_penalty = -0.15 * num_switches * max(0, (30 - cycles) / 30)

    utility = (
        match_up * 1.2 +                    # matchup
        my_hp_score - opp_hp_score +        # active pkm hp
        my_team_health * 0.8 +              # team health
        my_status_score - opp_status_score + # status conditions
        my_stage_score * 0.3 -              # stat stages
        opp_stage_score * 0.3 +             # opp stat stages
        weather_score +                      # weather effects
        switch_penalty                       # switch penalty
    )

    return utility

def evaluate_team_matchups(game_state: GameState) -> List[float]:
    my_team = game_state.teams[0]
    opp_active = game_state.teams[1].active
    weather = game_state.weather
    matchup_scores = []
    
    for pokemon in [my_team.active] + my_team.party:
        if pokemon.hp <= 0:  # Skip fainted Pokémon
            matchup_scores.append(float('-inf'))
            continue

        matchup = evaluate_actives_matchup(
            pokemon,
            opp_active
        )
        
        # consider weather benefits for this Pokémon
        if (weather == WeatherCondition.SUNNY and pokemon.type == PkmType.FIRE) or \
           (weather == WeatherCondition.RAIN and pokemon.type == PkmType.WATER):
            matchup += 0.5  # boost matchup score
        
        if (weather == WeatherCondition.SANDSTORM and pokemon.type == PkmType.ROCK):
            matchup += 0.3  # rock-types get Sp. Def boost

        if (weather == WeatherCondition.HAIL and pokemon.type == PkmType.ICE):
            matchup += 0.2  # ice-types are immune to Hail

        matchup_scores.append(matchup)
    
    return matchup_scores

class Node:
    def __init__(self, parent, connection_action):
        self.game_state: GameState = None
        self.parent = parent
        self.connection_action = connection_action
        self.visits = 0
        self.utility = 0.0
        self.children = []
    
    def is_fully_expanded(self):
        valid_action_count = sum(1 for i in range(DEFAULT_N_ACTIONS) 
                               if not is_action_suboptimal(self.game_state, i))
        return len(self.children) == valid_action_count
    
    def best_child(self):
        return max(self.children, key=lambda child: child.utility / (child.visits + 1e-6), default=None)

class MCTSPolicy(BattlePolicy):
    def __init__(self, max_iterations: int = 10_000, C:int = 10, debug = True, seed: int = 1):
        print('Starting MCTS Policy')
        self.max_iterations = max_iterations
        self.debug = debug
        self.turns = 0
        self.max_C = C*3
        self.C = C
        self.min_C = C
        random.seed(seed)
    
    def get_action(self, game_state: GameState):
        self.turns += 1
        
        self.C = max(self.min_C, self.max_C/(self.turns**1.2))
        
        print(f'Turn={self.turns}, C={self.C}')
        
        root = Node(None, None)
        root.game_state = deepcopy(game_state)
        
        action = self.mcts_search(root)
        return action
    
    def is_terminal(self, state): 
        return state.teams[1].active.hp == 0 or state.teams[0].active.hp == 0
    
    def is_switch(self, action):
        return action >= 4
        
    def playout(self, node):
        state = deepcopy(node.game_state)
        cycles = 0
        num_switches = 0
        while not self.is_terminal(state):
            matchup_scores = evaluate_team_matchups(state)
            current_score = matchup_scores[0]
            best_alternative = max(matchup_scores[1:], default=float('-inf'))
            
            valid_actions = []
            
            # if we have a significantly better matchup available, prioritize switching
            if (best_alternative - SWITCH_THRESHOLD) > current_score:
                # Find switch actions that improve our position
                for i in range(len(state.teams[0].active.moves), DEFAULT_N_ACTIONS):
                    switch_index = i - len(state.teams[0].active.moves)
                    if switch_index < len(matchup_scores) - 1:
                        if matchup_scores[switch_index + 1] > current_score:
                            valid_actions.append(i)
            
            # if no good switches found, consider moves that aren't suboptimal
            if not valid_actions:
                valid_actions = [i for i in range(DEFAULT_N_ACTIONS) 
                            if not is_action_suboptimal(state, i)]
            
            # if still no valid actions, allow all actions
            if not valid_actions:
                valid_actions = list(range(DEFAULT_N_ACTIONS))
                
            my_action = random.choice(valid_actions)                # i choose random i a list of optimized valid actions
            opp_action = random.randint(0, DEFAULT_N_ACTIONS - 1)   # opponent choose random: improve robustness
            
            if self.is_switch(my_action):
                num_switches += 1
            
            step_result = state.step([my_action, opp_action])
            if isinstance(step_result, tuple):
                next_states = step_result[0]
                if next_states:
                    state = next_states[0]
            cycles += 1
        return evaluate_terminal_state(state, cycles, num_switches)

    
    def back_propagate(self, node, utility):
        while node is not None:
            node.visits += 1
            node.utility += utility
            node = node.parent
    
    def ucb1(self, node):
        if node.visits == 0:
            return math.inf
        return node.utility / node.visits + self.C * math.sqrt(math.log(node.parent.visits + 1) / node.visits)
    
    def expand(self, node):
        if node.is_fully_expanded():
            return node.best_child()
        
        for i in range(DEFAULT_N_ACTIONS):
            if all(child.connection_action != i for child in node.children):
                # skip this action if it's a suboptimal move
                if is_action_suboptimal(node.game_state, i):
                    continue
                    
                new_node = Node(node, i)
                new_node.game_state = deepcopy(node.game_state)
                opp_action = random.randint(0, DEFAULT_N_ACTIONS - 1)
                step_result = new_node.game_state.step([i, opp_action])
                if isinstance(step_result, tuple):
                    next_states = step_result[0]
                    if next_states:
                        new_node.game_state = next_states[0]
                node.children.append(new_node)
                return new_node
    
    def mcts_search(self, root):
        for _ in (tqdm(range(self.max_iterations), desc='One turn simulations') if self.debug else range(self.max_iterations)):
            current = root
            # Selection
            while current.children and current.is_fully_expanded():
                current = max(current.children, key=self.ucb1)
            # Expansion
            if not current.is_fully_expanded():
                current = self.expand(current)
            # Simulation
            utility = self.playout(current)
            # Backpropagation
            self.back_propagate(current, utility)
        
        if len(root.children) == 0:        
            if self.debug:
                print('No children, going random')
            return random.randint(0, DEFAULT_N_ACTIONS - 1)
        
        best_action = max(root.children, key=lambda c: c.utility / (c.visits + 1e-6), default=None)
        
        if self.debug:
            if best_action:
                print(f"Best action: {best_action.connection_action}, Reward: {best_action.utility}, Visits: {best_action.visits}")
            else:
                print('No best action found, going random')
            
        return best_action.connection_action if best_action else random.randint(0, DEFAULT_N_ACTIONS - 1)
    