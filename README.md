# proximity learning

### Deprecated name - OWMM
#### Run these to update everything to the right stuff
```cd OWMM
git remote set-url origin git@github.com:Jdvakil/proximity_learning.git
cd .. && mv OWMM proximity_learning
./submodules/IsaacLab/isaaclab.sh --install
git fetch origin && git pull origin
```

Either create your branch in github UI or in cli with these:

```
git checkout -b <branch_name>
git push -u origin <branch-name>
```

### Install IsaacLab (run this before running any code in this repo)

`./isaaclab.sh --install`


### Run the code:

`cd submodules/IsaacLab/`

`./isaaclab.sh -p ../../sine_wave.py --enable_cameras --headless`

remove `--headless` to visualize the motions
