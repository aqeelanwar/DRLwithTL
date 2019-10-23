# Deep Reinforcement Learning with Transfer Learning - Simulated Drone and Environment (DRLwithTL-Sim)

## What is DRLwithTL-Sim?
This repository uses Transfer Learning (TL) based approach to reduce on-board computation required to train a deep neural network for autonomous navigation via Deep Reinforcement Learning for a target algorithmic performance. A library of 3D realistic meta-environments is manually designed using Unreal Gaming Engine and the network is trained end-to- end. These trained meta-weights are then used as initializers to the network in a **simulated** test environment and fine-tuned for the last few fully connected layers. Variation in drone dynamics and environmental characteristics is carried out to show robustness of the approach.
The repository containing the code for **real** environment on a **real** DJI Tello drone can be found @ [DRLwithTL-Real](https://github.com/aqeelanwar/DRLwithTL_real)


![Cover Photo](/images/cover.png)

![Cover Photo](/images/envs.png)


## Installing DRLwithTL-Sim
The current version of DRLwithTL-Sim supports Windows and requires python3. It’s advisable to [make a new virtual environment](https://towardsdatascience.com/setting-up-python-platform-for-machine-learning-projects-cfd85682c54b) for this project and install the dependencies. Following steps can be taken to download get started with DRLwithTL-Sim

### Clone the repository
```
git clone https://github.com/aqeelanwar/DRLwithTL.git
```

### Download imagenet weights for AlexNet
The DQN uses Imagenet learned weights for AlexNet to initialize the layers. Following link can be used to download the imagenet.npy file.

[Download imagenet.npy](https://drive.google.com/open?id=1Ei4mCzjfLY5ql6ILIUHaCtAR2XF6BtAM)

Once downloaded, place it in
```
models/imagenet.npy
```

### Install required packages
The provided requirements.txt file can be used to install all the required packages. Use the following command
```
cd DRLwithTL
pip install –r requirements.txt
```
This will install the required packages in the activated python environment.


### Install Epic Unreal Engine
You can follow the guidelines in the link below to install Unreal Engine on your platform

[Instructions on installing Unreal engine](https://docs.unrealengine.com/en-US/GettingStarted/Installation/index.html)

### Install AirSim
AirSim is an open-source plugin for Unreal Engine developed by Microsoft for agents (drones and cars) with physically and visually realistic simulations. In order to interface between Python3 and the simulated environment, AirSim needs to be installed. It can be downloaded from the link below

[Instructions on installing AirSim](https://github.com/microsoft/airsim)



## Running DRLwithTL-Sim
Once you have the required packages and software downloaded and running, you can take the following steps to run the code

### Create/Download a simulated environment
You can either manually create your environment using Unreal Engine, or can download one of the sample environments from the link below and run it.

* [Indoor Long Environment](https://drive.google.com/file/d/1yfFaI_9yXNa9iuLBbOtCfzoUOOV0iVoL/view?usp=sharing)

The link above will download the packaged version of the _Indoor Long environment_. Run the indoor_long.exe file to run the environment.

### Edit the configuration file (Optional)
The RL parameters for the DRL simulation can be set using the provided config file and are self-explanatory.

```
cd DRLwithTL\configs
notepad config.cfg (#for windows)
```

### Run the Python code
The DRL code can be started using the following command

```
cd DRLwithTL
python main.py
```

While the simulation is running, RL parameters such as epsilon, learning rate, average Q values and loss can be viewed on the tensorboard. The path depends on the env_type, env_name and train_type set in the config file and is given by 'models/trained/&lt;env_type>/&lt;env_name>/Imagenet/''. An example can be seen below

```
cd models\trained\Indoor\indoor_long\Imagenet\
tensorboard --logdir e2e

```


## Citing
If you find this repository useful for your research please use the following bibtex citations

```
@ARTICLE{2019arXiv191005547A,
       author = {{Anwar}, Aqeel and {Raychowdhury}, Arijit},
        title = "{Autonomous Navigation via Deep Reinforcement Learning for Resource Constraint Edge Nodes using Transfer Learning}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Machine Learning, Statistics - Machine Learning},
         year = "2019",
        month = "Oct",
          eid = {arXiv:1910.05547},
        pages = {arXiv:1910.05547},
archivePrefix = {arXiv},
       eprint = {1910.05547},
 primaryClass = {cs.LG},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2019arXiv191005547A},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
```
@article{yoon2019hierarchical,
  title={Hierarchical Memory System With STT-MRAM and SRAM to Support Transfer and Real-Time Reinforcement Learning in Autonomous Drones},
  author={Yoon, Insik and Anwar, Malik Aqeel and Joshi, Rajiv V and Rakshit, Titash and Raychowdhury, Arijit},
  journal={IEEE Journal on Emerging and Selected Topics in Circuits and Systems},
  volume={9},
  number={3},
  pages={485--497},
  year={2019},
  publisher={IEEE}
}
```

## Authors
* [Aqeel Anwar](https://www.prism.gatech.edu/~manwar8) - Georgia Institute of Technology

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details
