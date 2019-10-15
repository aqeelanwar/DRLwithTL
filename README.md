# Deep Reinforcement Learning with Transfer Learning - Simulated Drone and Environment (DRLwithTL-Sim)

## What is DRLwithTL-Sim?
This repository uses Transfer Learning (TL) based approach to reduce on-board computation required to train a deep neural network for autonomous navigation via Deep Reinforcement Learning for a target algorithmic performance. A library of 3D realistic meta-environments is manually designed using Unreal Gaming Engine and the network is trained end-to- end. These trained meta-weights are then used as initializers to the network in a **simulated** test environment and fine-tuned for the last few fully connected layers. Variation in drone dynamics and environmental characteristics is carried out to show robustness of the approach.
The repository containing the code for **real** environment on a **real** DJI Tello drone can be found @ [DRLwithTL-Real](www.google.com)

## Installing DRLwithTL-Sim
The current version of DRLwithTL-Sim supports the most commonly used OS such as Windows, Linux and Mac OS and requires python3. It’s advisable to make a new virtual environment for this project and install the dependencies. Following steps can be taken to download get started with DRLwithTL-Sim

### Clone the repository
```
git clone https://github.com/aqeelanwar/DRLwithTL.git
```
### Install required packages
The provided requirements.txt file can be used to install all the required packages. Use the following command

### Install Epic Unreal Engine

### Run the simulated environment
AirSim is used to interface between the Python code and Unreal Engine simulated environments. You can either use Unreal Engine to manually design the environment or you can download one from the link below

* [Indoor Long Environment](https://www.google.com)

## Citing
If you find this repository useful for your research please use the following bibtex citation

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

## Authors
* [Aqeel Anwar](https://www.prism.gatech.edu/~manwar8) - Georgia Institute of Technology

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details
