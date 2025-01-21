from vgc.behaviour.BattlePolicies import Minimax, TypeSelector, TunedTreeTraversal, PrunedBFS
from competitor_wrapper import CompetitorWrapper
from mcts import MCTSPolicy
from vgc.competition.BattleMatch import BattleMatch
from vgc.competition.Competitor import CompetitorManager
from vgc.util.generator.PkmRosterGenerators import RandomPkmRosterGenerator
from vgc.util.generator.PkmTeamGenerators import RandomTeamFromRoster 

competitor0 = CompetitorWrapper('Player0')
competitor1 = CompetitorWrapper('Player1')

competitor0._battle_policy = MCTSPolicy(max_iterations=1_000, debug=True)
competitor1._battle_policy = TypeSelector() # choose here the opponent policy: Minimax, TypeSelector, TunedTreeTraversal, PrunedBFS

cm0 = CompetitorManager(competitor0)
cm1 = CompetitorManager(competitor1)

roster = RandomPkmRosterGenerator().gen_roster()
team0 = RandomTeamFromRoster(roster).get_team()
team1 = RandomTeamFromRoster(roster).get_team()

cm0.team = team0
cm1.team = team1
    
match = BattleMatch(cm0, cm1, debug=True)
match.run()

print(f'You...{"WON! 🎉" if match.winner() == 0 else "loose...😨"}')