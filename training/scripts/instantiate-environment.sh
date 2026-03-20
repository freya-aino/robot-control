sudo apt-get install git-lfs
git lfs install
git clone --recurse https://github.com/freya-aino/unitree-robot.git
cd unitree-robot
pip install uv
uv venv
uv sync --no-dev