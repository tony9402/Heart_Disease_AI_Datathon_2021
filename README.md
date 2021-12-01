# Heart Disease AI Datathon 2021

### Clone

```
git clone https://github.com/tony9402/Heart_Disease_AI_Datathon_2021
cd Heart_Disease_AI_Datathon_2021
```

### Build Dockerfile

```
docker build . -t hdad_docker
```

### Run Docker and attach

```
docker run -itd --name HDAD --gpus all --net=host --ipc=host -v $(pwd):/github hdad_docker
docker attach HDAD
```

### Preprocessing Data

```
TODO
```

### File Hierarchy

```
github
├─  training
└─  data
```
