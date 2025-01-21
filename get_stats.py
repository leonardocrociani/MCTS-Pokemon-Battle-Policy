# get the stats of the different opponents

import os 

if not os.path.exists('./outputs'):
    if not os.path.exists('./outputs'):
        print('No outputs found')
        exit(1)

for i in ['type_selector', 'pruned_bfs', 'minimax', 'tuned_tree']:
    filename = f'./outputs/{i}.txt'
    with open(filename, 'r') as f:
        content = f.read()
    
    turns = content.count('TURN')
    fainted = content.count('FAINTED')
    tot_battles = content.count('BATTLE')
    won = content.count('0 Won')
    
    print(f'Opponent: {i}')
    print(f'Turns: {turns/tot_battles}')
    print(f'Fainted: {fainted/tot_battles}')
    print(f'Total battles: {tot_battles}')
    print(f'Win rate: {won/tot_battles}\n')
    