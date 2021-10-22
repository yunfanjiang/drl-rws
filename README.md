# cs238-rps
CS238 Final Project Running with Scissors

## Setup
Install dependencies

`git clone https://github.com/yunfanjiang/meltingpot && cd meltingpot`

`pip install -e .`

Install this repo as develop mode

`git clone https://github.com/yunfanjiang/cs238-rps && cd cs238-rps`

`pip install -e .`

## Examples
### Random Agent
`python examples/random_rps.py`

### Simple PPO
`python examples/simple_ppo.py`

## Sample colab setup

### Get customized meltingpot repo:
`!git clone https://github.com/yunfanjiang/meltingpot`

`%cd meltingpot/`

`pip install -e .`

### Get our RPS repo and install dependencies:
`!git clone https://<github-token>@github.com/yunfanjiang/cs238-rps.git`

`%cd ../cs238-rps/`

`pip install -e .`

`pip install pip/*`

`pip install -r requirements.txt`

### Example PPO Training
`!python examples/simple_ppo.py`