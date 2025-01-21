#!/bin/bash

if [ ! -d "pokemon-vgc-engine" ]; then
    git clone https://gitlab.com/DracoStriker/pokemon-vgc-engine.git
    pushd pokemon-vgc-engine && git reset --hard 2f23335a8f6796d46b4aabfb163c0f17c0dc5fe6 && popd
fi

if [ ! -d "venv" ]; then
    python3.11 -m venv venv
fi

source venv/bin/activate

pip install ./pokemon-vgc-engine
pip install -r requirements.txt

read -p "Do you want to run a battle? (y/n): " choice
if [[ "$choice" == "y" ]]; then
    python single-battle.py
fi