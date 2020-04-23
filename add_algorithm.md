# Steps to add new detector / tracker / ReID Algorithm

1. Select a unique algorithm name and enter into the following key in config.ini
* For detector or tracker algorithm
```
det_algorithm=<NEW_ALGORITHM>
```
* For ReID algorithm
```
reid_algorithm=<NEW_ALGORITHM>
```

2. Below the [General] section in config.ini, add a new section for the new algorithm. The name of the new section must match the new algorithm name.
```
[General]
...
...

[<NEW_ALGORITHM>]
...
...
```

3. Under the newly created section, enter variables required to instantiate or for options for the new algorithm.
```
[<NEW_ALGORITHM>]
variable1=<OPTION1>
variable2=<OPTION2>
```

4. Create a wrapper class for the new algorithm and place them in their dedicated folders.
* For detector or tracker algorithm, create a new folder for the new algorithm under the ```Detector_Tracker``` folder and place all relevant files/models/weights/etc inside.
* For ReID algorithm, create a new folder for the new algorithm under the ```ReID``` folder and place all relevant files/models/weights/etc inside/

5. Open ```Algorithm_Factory.py``` and import the wrapper class for the new algorithm.

6. Add code to instantiate the new algorithm in their respective algorithm methods.
* For detector or tracker algorithm
```
def Detector_Tracker_Factory(config):
    detector = None
    algo = config['det_algorithm']
    ...
    ...
    if algo = '<NEW_ALGORITHM>':
        detector = NEW_ALGORITHM(
            ...
        )
    return detector
```
* For ReID algorithm
```
def ReID_Factory(config):
    reid = None
    algo = config['reid_algorithm']
    ...
    ...
    if algo = '<NEW_ALGORITHM>':
        reid = NEW_ALGORITHM(
            ...
        )
    return reid
```

7. [For ReID Algorithm only] Open ```MultiVideoProcessor.py``` and go to method ```_reid_process_factory(self, reid)```. Add a new entry to the factory dictory key and assign the value to a new function for processing ReID feature extraction and scoring (refer to ```_processPersonReid``` method for example)
```
def _reid_process_factory(self, reid):
    factory = {
        '<NEW_ALGORITHM>': <NEW_ALGO_PROCESS_REID_METHOD>
    }
    return factory[reid]
```