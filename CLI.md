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


cd /media/michael/datafiles/Work-syncfree/IsaacLab

./isaaclab.sh -p -m pip uninstall rsl-rl-lib -y
rm -rf _isaac_sim/kit/python/lib/python3.10/site-packages/rsl_rl
rm -rf _isaac_sim/kit/python/lib/python3.11/site-packages/rsl_rl

cd SRU_rsl_rl
../isaaclab.sh -p -m pip install -e .

../isaaclab.sh -p -c "from rsl_rl.modules import ActorCriticSRU; print('✓ SRU modules loaded')"


cd /media/michael/datafiles/Work-syncfree/IsaacLab
./isaaclab.sh -p -m pip install -e source/isaaclab_nav_task

./isaaclab.sh -p -m pip show isaaclab_nav_task
./isaaclab.sh -p source/isaaclab_nav_task/scripts/train.py --help


./isaaclab.sh -p source/isaaclab_nav_task/scripts/play.py \
    --task Isaac-Nav-PPO-B2W-Play-v0 \
    --num_envs 1 \
    --checkpoint source/isaaclab_nav_task/logs/rsl_rl/b2w_navigation_ppo/2026-02-10_10-03-38/model_3000.pt








python scripts/list_envs.py

# sanity check
python scripts/play.py Unitree-Go2-DreamWaq --agent random --viewer=viser 
python scripts/play.py Unitree-Go2-Rough --agent random --viewer=viser 

# train
python scripts/train.py Unitree-Go2-DreamWaq --env.scene.num-envs=4096

# play
python scripts/play.py Unitree-Go2-DreamWaq --viewer=viser --checkpoint_file=logs/rsl_rl/go2_dreamwaq/2026-03-24_17-38-51/model_0.pt

tensorboard --logdir_spec \
dreamwaq:logs/rsl_rl/go2_dreamwaq,\
5090:logs/rsl_rl/5090_go2_dreamwaq,\
isaac_gym:logs/isaac_gym_logs/rough_go2_waq/Mar14_19-20-17_waq_10k

tensorboard --logdir_spec \
Go2:logs/rsl_rl/go2_velocity,\
Xiaoli:logs/rsl_rl/xiaoli_velocity

nvidia-smi -l 1
