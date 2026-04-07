# Isaac Sim root directory
export ISAACSIM_PATH="/media/michael/datafiles/isaac-sim"
# Isaac Sim python executable
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
${ISAACSIM_PATH}/isaac-sim.sh
# checks that python path is set correctly
${ISAACSIM_PYTHON_EXE} -c "print('Isaac Sim configuration is now complete.')"
# checks that Isaac Sim can be launched from python
${ISAACSIM_PYTHON_EXE} ${ISAACSIM_PATH}/standalone_examples/api/isaacsim.core.api/add_cubes.py

# enter the cloned repository
cd IsaacLab
# create a symbolic link
ln -s ${ISAACSIM_PATH} _isaac_sim

sudo apt install cmake build-essential
### Install isaaclab
Run the install command that iterates over all the extensions in source directory and installs them using pip (with --editable flag):
./isaaclab.sh --install # or "./isaaclab.sh -i"

> this has potential issue after copy SRU_rsl_rl and source/isaaclab_nav_task

### SRU_rsl_rl install
SRU task has its own customized rsl-rl, we need to replace the pre-installed rsl_rl package:
./isaaclab.sh -p -m pip uninstall rsl-rl-lib -y
rm -rf _isaac_sim/kit/python/lib/python3.10/site-packages/rsl_rl
rm -rf _isaac_sim/kit/python/lib/python3.11/site-packages/rsl_rl
rm -rf _isaac_sim/kit/python/lib/python3.12/site-packages/rsl_rl

cd SRU_rsl_rl
../isaaclab.sh -p -m pip install -e .
#### Verify SRU_rsl_rl installation
../isaaclab.sh -p -c "from rsl_rl.modules import ActorCriticSRU; print('✓ SRU modules loaded')"

### isaaclab_nav_task install
cd /media/michael/datafiles/Work-syncfree/IsaacLab
./isaaclab.sh -p -m pip install -e source/isaaclab_nav_task

#### Verify isaaclab_nav_task installation
./isaaclab.sh -p -m pip show isaaclab_nav_task
./isaaclab.sh -p source/isaaclab_nav_task/scripts/train.py --help

./isaaclab.sh -p -c "import gymnasium as gym; import isaaclab_nav_task; print([id for id in gym.registry.keys() if 'Isaac-Nav-'in id])"
./isaaclab.sh -p scripts/environments/list_envs.py


### Install isaaclab
isaaclab --install

> this has potential issue after copy SRU_rsl_rl and source/isaaclab_nav_task

### SRU_rsl_rl install
#SRU task has its own customized rsl-rl, we need to replace the pre-installed rsl_rl package:
python -m pip uninstall rsl-rl-lib -y
rm -rf _isaac_sim/kit/python/lib/python3.10/site-packages/rsl_rl
rm -rf _isaac_sim/kit/python/lib/python3.11/site-packages/rsl_rl
rm -rf _isaac_sim/kit/python/lib/python3.12/site-packages/rsl_rl

cd SRU_rsl_rl
python -m pip install -e .
#### Verify SRU_rsl_rl installation
python -c "from rsl_rl.modules import ActorCriticSRU; print('✓ SRU modules loaded')"

### isaaclab_nav_task install
cd /media/michael/datafiles/Work-syncfree/IsaacLab
python -m pip install -e source/isaaclab_nav_task

#### Verify isaaclab_nav_task installation
python -m pip show isaaclab_nav_task
python source/isaaclab_nav_task/scripts/train.py --help

python -c "import gymnasium as gym; import isaaclab_nav_task; print([id for id in gym.registry.keys() if 'Isaac-Nav-'in id])"
python scripts/environments/list_envs.py

# train
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-Unitree-Go2-SRU-v0 \
  --num_envs 4096 --max_iterations 30000 --headless --device cpu


python source/isaaclab_nav_task/scripts/play.py \
    --task Isaac-Nav-PPO-B2W-Play-v0 \
    --num_envs 1 \
    --checkpoint source/isaaclab_nav_task/logs/rsl_rl/b2w_navigation_ppo/2026-02-10_10-03-38/model_3000.pt


cat >> .venv/bin/activate <<'EOF'
export ISAACLAB_PATH="/media/michael/datafiles/Work-syncfree/IsaacLab"
alias isaaclab="${ISAACLAB_PATH}/isaaclab.sh"
export RESOURCE_NAME="IsaacSim"
if [ -f "${ISAACLAB_PATH}/_isaac_sim/setup_conda_env.sh" ];then
    . "${ISAACLAB_PATH}/_isaac_sim/setup_conda_env.sh"
fi
EOF

tensorboard --logdir_spec \
Go2:logs/rsl_rl/go2_velocity,\
Xiaoli:logs/rsl_rl/xiaoli_velocity

nvidia-smi -l 1


ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
nvidia-srl-usd 2.0.0 requires usd-core<26.0,>=25.2.post1; python_version >= "3.11", which is not installed.
fastapi 0.115.7 requires starlette<0.46.0,>=0.40.0, but you have starlette 0.49.1 which is incompatible.

python -m pip uninstall -y usd-core
python -m pip install "usd-core>=25.2.post1,<26.0"
python -m pip uninstall -y starlette
python -m pip install "starlette>=0.40.0,<0.46.0"
python -m pip install nvidia-srl-usd
python -m pip install fastapi



SRU_rsl_rl 和 source/isaaclab_nav_task 是新复制加进来的文件夹，它们共同服务sru项目在Isaac           
Lab训练.目前源码是b2w机器人，我打算移植go2机器人到sru，首先需要locomotion才能开始训练，帮我使用统一的 
go2资产和配置，训练go2的locomotion。注意isaac lab自带的locomotion任务使用的是原版rsl，但是sru的rsl是  
修改过的，需要使用sru版本确保后续训练的兼容性。可以参考b2w的源码，尽量对齐b2w的代码但是要注意isaaclab 
和isaacsim的版本差异。目前只做go2rough地形的低层训练，创建新配置，不要修改isaac源码 