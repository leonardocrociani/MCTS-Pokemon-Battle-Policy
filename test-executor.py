from mcts import MCTSPolicy

from vgc.competition.BattleMatch import BattleMatch
from vgc.competition.Competitor import CompetitorManager
from vgc.util.generator.PkmRosterGenerators import RandomPkmRosterGenerator
from vgc.util.generator.PkmTeamGenerators import RandomTeamFromRoster 
from tqdm import tqdm

from vgc.behaviour.BattlePolicies import Minimax, TypeSelector, TunedTreeTraversal, PrunedBFS
from competitor_wrapper import CompetitorWrapper
import sys

args = sys.argv

opponent_policies = {
  'tuned_tree' : TunedTreeTraversal(),
  'minimax': Minimax(),
  'type_selector': TypeSelector(),
  'pruned_bfs': PrunedBFS(),
}

if not args[1] in opponent_policies:
  print('Invalid opponent name')
  sys.exit(1)

opponent_name = args[1]
opponent_policy = opponent_policies[opponent_name]

NUM_TESTS_PER_OPPONENT = 1
ith_test = 0

competitor0 = CompetitorWrapper('Player0')
competitor1 = CompetitorWrapper('Player1')

competitor0._battle_policy = MCTSPolicy(max_iterations=10_000, debug=False)
competitor1._battle_policy = opponent_policy

cm0 = CompetitorManager(competitor0)
cm1 = CompetitorManager(competitor1)

for _ in tqdm(range(NUM_TESTS_PER_OPPONENT), desc=f'Testing {opponent_name} opponent'):
  roster = RandomPkmRosterGenerator().gen_roster()
    
  team0 = RandomTeamFromRoster(roster).get_team()
  team1 = RandomTeamFromRoster(roster).get_team()
  
  cm0.team = team0
  cm1.team = team1
    
  match = BattleMatch(cm0, cm1, debug=True)
  match.run()
  
  print(f'Intermediate test {opponent_name}, {ith_test} test...: won?', 'YES!' if match.winner() == 0 else 'NO!')
  
  cm1.team = team0
  cm0.team = team1
    
  match = BattleMatch(cm1, cm0, debug=True)
  match.run()

  print(f'Test {ith_test} completed with {opponent_name} opponent, won?', 'YES!' if match.winner() == 0 else 'NO!')

  ith_test += 1 