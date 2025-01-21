# Hi! 👋🏻

Here you can find a MCTS-based algorithm guided by heuristics to solve the pokemom battle games.

## How to run the code

I've tested the code with python 3.11, make sure you have it installed.

To run the code, just execute:

```bash
./setup.sh
```

The script will install the required dependencies and ask you if you want to run a battle!

Tune your parameter inside the `single-battle.py` file.

## Other files

- `AIF*.ipynb`: The notebooks I used to submit the code for the exam.
- `mcts.py`: Contains the MCTS battle policy.
- `run-tests.sh`: Script to run the tests.
- `kill-engine.sh`: Script to run kill the engine (the fastest way).
- `get-stats.py`: Script to retrieve the stats from the battles (run it after all the battles of the tests terminated).
- `competitor-wrapper.py`: A utility class.

Make sure to activate the virtul environment `venv` if you want to run the code in a different way or to run the tests.